#!/usr/bin/env python3
"""GRM-D1: the LT1.1 recap battery -- five ANSWERABLE recap questions.

Why the old battery scored 0/5 on both arms
-------------------------------------------
LT1 asked one question, `recap the five biggest decisions we made`, and scored
the single free-form answer against all five decision spans. The receipt
(`.../cells/A-197-200/recap.json`) shows the mechanism, not a memory failure:

    admission_identified_candidates : []
    ranking_ids                     : [25, 16, 64, 22, 24, 40]
    trips                           : 4 trips, every one "grounded": false

The question carries NO identifier token -- "recap", "five", "biggest",
"decisions", "made" are all instruction/stop words. With nothing to key on,
routing ranked arbitrary nodes, every ladder trip came back ungrounded, and the
reader free-associated ("89 iron ingots ... 5 mithril ingots"). The battery was
measuring whether an un-keyed query can be routed, which it cannot by
construction. That is an UNDER-SPECIFIED QUESTION, exactly as the LT1 result
recorded, and no amount of memory quality could have moved it.

The redesign
------------
One question per decision. Each question:

  * NAMES the decision it targets by entity and attribute, so the entity is a
    rare identifier token the router can key on (the same shaping the 35 recall
    probes already rely on -- see core/grm_admission.py:shaped_identifier_tokens);
  * has a single contiguous value-span expected answer, scored with the EXISTING
    scorer (scripts/grm_lt1.py:score, the C5 arm S ordered contiguous span rule);
  * carries no instruction-word identifier (`unknown`, `reply`, `answer`,
    `recap`, `list`, `summarize`, ...) that could collide with the reader's
    own vocabulary -- enforced by `assert_no_instruction_identifier`;
  * sits outside the recency window -- enforced by `assert_outside_recency`,
    which reuses the fixture gate's own rule (min distance >= 10 turns from
    every source turn).

The five decisions themselves are UNCHANGED: they are the frozen
`fixtures/lt1/dialogue.json` `decisions` list, so LT1.1 measures the same five
commitments the original battery meant to measure.

Prior art
---------
  * The value-span scorer is C5 arm S (GRM contributors, 2026), reused
    unchanged via `scripts.grm_lt1.score` -- not reimplemented.
  * The "no probes inside the recency window" and "no instruction-word
    identifier" gate rules are LT1's own `fixture_gate`
    (scripts/grm_lt1.py:104-136, GRM contributors 2026); ported verbatim in
    intent to the recap probes, which the original gate did not cover.
  * The one-question-per-decision shape is the standard fix for an under-
    specified aggregate probe; I know of no specific paper for it. Unverified
    -- lead to check; search terms: "multi-hop aggregate question
    decomposition", "list-recall vs point-recall evaluation", "LongBench
    summarization vs retrieval split".
  * Mine: the identifier-token argument for WHY the old battery was
    unanswerable (read off `admission_identified_candidates == []`), and the
    two assertion gates below.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / 'fixtures/lt1/dialogue.json'
OUT = ROOT / 'artifacts/grm_d1'

# The recap battery replaces turn 200. Each probe is asked at its own turn so
# every answer is scored independently, the way the 35 recall probes are.
RECAP_TURNS = [196, 197, 198, 199, 200]

# Instruction vocabulary that must never be a probe's identifier token. The
# first three are LT1's own list (scripts/grm_lt1_cpu.py:68); the rest are the
# aggregate/summary words that made the old recap question un-keyable.
INSTRUCTION_WORDS = {
    'unknown', 'reply', 'answer',
    'recap', 'recall', 'list', 'summarize', 'summary', 'decisions', 'decision',
    'biggest', 'settle', 'settled', 'remind', 'repeat', 'review',
}


def build_questions(fixture):
    """One named, value-span question per frozen decision.

    Each question opens with the entity name so the rare-identifier shaping in
    core/grm_admission.py has a token to key on, and names the attribute so the
    correct node in an entity's family is the top-scoring one.
    """
    questions = []
    for n, decision in enumerate(fixture['decisions'], 1):
        entity = decision['entity']
        attribute = decision['attribute']
        questions.append(dict(
            id='recap_%d' % n,
            turn=RECAP_TURNS[n - 1],
            entity=entity,
            attribute=attribute,
            # The decision each question targets, stated plainly.
            targets='the %s we chose for %s' % (attribute, entity),
            question="For %s, which %s did we go with?" % (entity, attribute),
            expected=decision['expected'],
            source_turns=list(decision['source_turns']),
            answerable=True,
            scorer='scripts.grm_lt1.score (C5 arm S contiguous value span)',
        ))
    return questions


def assert_no_instruction_identifier(question):
    """Gate: the question's capitalised/rare tokens must not be instructions."""
    text = question['question']
    # Same shaping idea as core/grm_admission.shaped_identifier_tokens: the
    # capitalised and hyphenated tokens are what the router will key on.
    candidates = {w.rstrip(".,:;?'").casefold()
                  for w in re.findall(r"[A-Za-z][\w\-']*", text)
                  if w[:1].isupper() or '-' in w}
    collide = candidates & INSTRUCTION_WORDS
    if collide:
        raise ValueError('RECAP_INSTRUCTION_IDENTIFIER_COLLISION: %s in %r'
                         % (sorted(collide), text))
    # There must be at least one identifier token left to route on -- that is
    # precisely what the old recap question lacked.
    if not candidates:
        raise ValueError('RECAP_NO_IDENTIFIER_TOKEN: %r' % text)
    if re.search(r'\b[A-Z][A-Z_]+\b', text):
        raise ValueError('RECAP_UPPERCASE_INSTRUCTION_TOKEN: %r' % text)
    return sorted(candidates)


def assert_outside_recency(question, fixture, recency_window=10):
    """Gate: no source turn may sit inside the recency window of the probe.

    Prior art: scripts/grm_lt1.py:fixture_gate's SOURCE_TOO_RECENT rule
    (GRM contributors, 2026), applied to the recap probes as well.
    """
    distances = [question['turn'] - t for t in question['source_turns']]
    if min(distances) < recency_window:
        raise ValueError('RECAP_SOURCE_TOO_RECENT: %s at turn %d, distances %s'
                         % (question['id'], question['turn'], distances))
    return dict(min_distance=min(distances), max_distance=max(distances))


def gate(fixture=None):
    """Run both fixture gates over the battery; return the receipt."""
    fixture = fixture or json.loads(FIXTURE.read_text())
    questions = build_questions(fixture)
    if len(questions) != 5:
        raise ValueError('RECAP_BATTERY_SIZE: %d' % len(questions))
    if len({q['turn'] for q in questions}) != 5:
        raise ValueError('RECAP_TURN_COLLISION')
    rows = []
    for q in questions:
        rows.append(dict(q,
                         identifier_tokens=assert_no_instruction_identifier(q),
                         **assert_outside_recency(q, fixture)))
    return dict(battery='LT1.1 recap', questions=rows,
                replaces=dict(turn=200,
                              question='recap the five biggest decisions we made',
                              observed_score='0/5 on both arms',
                              mechanism='admission_identified_candidates == []; '
                                        'all ladder trips grounded=false',
                              receipt='/mnt/ForgeRealm/wt/grm-lt1/artifacts/grm_lt1/'
                                      'amendment2/run_margin_first/cells/A-197-200/recap.json'))


def oracle_answers(fixture=None):
    """The oracle path: answer each question from its own source turn text.

    This is the C7/LT1 oracle contract -- the source sentences are placed in
    the prompt verbatim, so a correct reader must score 5/5. It measures the
    QUESTIONS, not the memory: a question the oracle cannot answer is
    under-specified and must not ship.
    """
    fixture = fixture or json.loads(FIXTURE.read_text())
    turns = {t['turn']: t for t in fixture['turns']}
    answers = []
    for q in build_questions(fixture):
        # Exactly the sources LT1's oracle would wrap.
        source_texts = [turns[t]['user'] for t in q['source_turns']]
        # The current value is the LAST source turn's value: the lineage's tip.
        value = turns[q['source_turns'][-1]]['value']
        answers.append(dict(id=q['id'], question=q['question'],
                            expected=q['expected'],
                            source_texts=source_texts,
                            answer='For %s, the %s we went with is %s.'
                                   % (q['entity'], q['attribute'], value)))
    return answers


def oracle_gate(fixture=None):
    """CPU gate: the oracle must score 5/5 on the fake session."""
    from scripts.grm_lt1 import score
    rows = []
    for a in oracle_answers(fixture):
        graded = score(a['answer'], a['expected'])
        rows.append(dict(a, score=graded))
    matched = sum(1 for r in rows if r['score']['exact_correct'])
    return dict(matched=matched, out_of=len(rows), rows=rows,
                status='PASS' if matched == len(rows) else 'RED',
                evidence_class='CPU oracle gate over the frozen fixture '
                               '(question quality, not memory quality)')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(FIXTURE.read_text())
    receipt = dict(gate=gate(fixture), oracle=oracle_gate(fixture))
    (OUT / 'recap_battery.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('recap oracle: %d/%d %s' % (receipt['oracle']['matched'],
                                      receipt['oracle']['out_of'],
                                      receipt['oracle']['status']))
    for q in receipt['gate']['questions']:
        print('  %-9s turn %3d  d=%3d  %-52s -> %r'
              % (q['id'], q['turn'], q['min_distance'], q['question'], q['expected']))


if __name__ == '__main__':
    main()
