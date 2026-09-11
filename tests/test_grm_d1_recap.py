"""GRM-D1: gates for the LT1.1 recap battery.

The old battery was one un-keyable question scored against five spans and
returned 0/5 on both arms. These tests hold the replacement to the properties
that made the old one unmeasurable, and prove each gate can go RED.

Prior art: the gate rules are LT1's own `fixture_gate`
(scripts/grm_lt1.py:104-136, GRM contributors 2026) and the C5 arm S value-span
scorer, both reused rather than reimplemented. See scripts/grm_d1_recap.py for
the full prior-art note.
"""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_recap as recap  # noqa: E402
from scripts.grm_lt1 import score  # noqa: E402


@pytest.fixture(scope='module')
def fixture():
    return json.loads(recap.FIXTURE.read_text())


def test_battery_is_five_named_answerable_questions(fixture):
    questions = recap.build_questions(fixture)
    assert len(questions) == 5
    decisions = fixture['decisions']
    for q, d in zip(questions, decisions):
        # Each question NAMES the decision it targets.
        assert q['entity'] == d['entity']
        assert q['attribute'] == d['attribute']
        assert q['entity'] in q['question']
        assert q['attribute'] in q['question']
        # ... and carries a value-span expected answer.
        assert q['expected'] == d['expected']
        assert q['answerable'] is True
        assert q['source_turns'] == list(d['source_turns'])


def test_expected_answers_are_scored_by_the_existing_value_span_scorer(fixture):
    """The scorer is LT1's, unchanged: a right answer scores, a wrong one does not."""
    for q in recap.build_questions(fixture):
        right = 'We went with %s.' % q['expected']
        assert score(right, q['expected'])['category'] == 'correct'
        wrong = 'We went with something else entirely.'
        assert score(wrong, q['expected'])['category'] != 'correct'


def test_no_probe_sits_inside_the_recency_window(fixture):
    for q in recap.build_questions(fixture):
        span = recap.assert_outside_recency(q, fixture)
        assert span['min_distance'] >= 10


def test_recency_gate_can_go_red(fixture):
    """RED proof for the recency rule."""
    q = dict(recap.build_questions(fixture)[0])
    q['turn'] = q['source_turns'][-1] + 3          # inside the window
    with pytest.raises(ValueError, match='RECAP_SOURCE_TOO_RECENT'):
        recap.assert_outside_recency(q, fixture)


def test_no_question_carries_an_instruction_word_identifier(fixture):
    for q in recap.build_questions(fixture):
        tokens = recap.assert_no_instruction_identifier(q)
        assert tokens, 'every question must leave a routable identifier'
        assert not set(tokens) & recap.INSTRUCTION_WORDS


def test_instruction_identifier_gate_can_go_red(fixture):
    """RED proof: the OLD recap question fails this gate outright."""
    old = dict(question='Recap the five biggest decisions we made')
    with pytest.raises(ValueError, match='RECAP_INSTRUCTION_IDENTIFIER_COLLISION'):
        recap.assert_no_instruction_identifier(old)
    # And a question with no identifier at all is rejected too.
    with pytest.raises(ValueError, match='RECAP_NO_IDENTIFIER_TOKEN'):
        recap.assert_no_instruction_identifier(dict(question='what did we choose?'))


def test_oracle_answers_five_of_five_on_the_fake_session(fixture):
    """The registration gate: a question the oracle cannot answer must not ship."""
    result = recap.oracle_gate(fixture)
    assert result['status'] == 'PASS'
    assert result['matched'] == 5
    assert result['out_of'] == 5
    for row in result['rows']:
        assert row['score']['exact_correct'] is True
        # The oracle sees the source sentences verbatim, per the C7/LT1 contract.
        assert row['source_texts']


def test_oracle_gate_reports_red_when_a_question_is_unanswerable(fixture):
    """RED proof for the oracle gate itself."""
    broken = json.loads(json.dumps(fixture))
    broken['decisions'][0]['expected'] = 'a value that is nowhere in the dialogue'
    result = recap.oracle_gate(broken)
    assert result['status'] == 'RED'
    assert result['matched'] < result['out_of']


def test_correction_lineage_decisions_expect_the_CURRENT_value(fixture):
    """recap_4 / recap_5 target correction families: the tip, not the original."""
    questions = {q['id']: q for q in recap.build_questions(fixture)}
    assert questions['recap_4']['expected'] == '29 credits'   # not 17 / 23
    assert questions['recap_5']['expected'] == '16 beds'      # not 12 / 14
    assert len(questions['recap_4']['source_turns']) == 3
    assert len(questions['recap_5']['source_turns']) == 3


def test_gate_receipt_names_what_it_replaces(fixture):
    receipt = recap.gate(fixture)
    replaced = receipt['replaces']
    assert replaced['question'] == 'recap the five biggest decisions we made'
    assert replaced['observed_score'] == '0/5 on both arms'
    assert 'admission_identified_candidates == []' in replaced['mechanism']
    assert replaced['receipt'].endswith('A-197-200/recap.json')
    assert len(receipt['questions']) == 5
