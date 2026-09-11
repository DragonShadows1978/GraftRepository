"""GRM-D1: CPU proof that LT1's corrections are a fixture-lineage trap.

Two tests, one contrast:

  * `test_feed_only_keeps_stale_node_active_and_first` reproduces the LT1
    harness exactly (`arena.feed()` for every turn, including the correction
    turns, scripts/grm_lt1_worker.py:147) and shows the stale node is STILL an
    eligible route candidate and still outranks the current one -- the exact
    shape of the 5 wrong `recall_4_*` rows.
  * `test_supersede_retires_old_value_and_ranks_new_first` applies the same
    three turns through the production correction path
    (`repo.apply_memory_command('correct memory: <old> => <new>')`, the path
    `scripts/grm_e2e_session.py:2590-2597` uses for `kind == "supersede"`) and
    shows the old nodes are retired, dropped from `_route_cand_base`, and the
    current value ranks first and is the only value reachable.

The point is NOT a language-model quality claim: the reader here is the C7 CPU
prose double. The claim is about NODE LINEAGE and ROUTE ELIGIBILITY only.

Prior art:
  * C7 r3 (`wt/grm-c7` scripts/grm_c7_register_r3.py:41-49) originated the
    `kind='supersede'` + `correction_command` fixture shape and the
    `correct memory: <old> => <new>` command string. Taken verbatim as the
    shape to port into LT1. Mine: the LT1-specific feed-only contrast arm and
    the eligibility assertion against `_route_cand_base`.
  * `correct_memory` / `_route_cand_base` are production GRM code
    (GRM contributors, 2026), used unchanged; nothing in core is modified.
  * No prior art known to me for this exact paired CPU contrast.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_d1_supersession import (  # noqa: E402
    CORRECTION_TURNS, feed_only_arm, supersede_arm,
)


def test_feed_only_keeps_stale_node_active_and_first(tmp_path):
    """The LT1 trap, reproduced: nothing is retired, stale still ranks first."""
    result = feed_only_arm(tmp_path / 'feed_only')

    # Every correction turn is still an ACTIVE, eligible route candidate.
    assert result['retired_nodes'] == [], (
        'feed-only must retire nothing; got %r' % result['retired_nodes'])
    assert set(result['family_nodes']).issubset(set(result['route_cand_base'])), (
        'every correction-family node stays route-eligible under feed-only')

    # And the STALE node, not the current one, heads the ranking.
    assert result['ranking'], 'the probe must rank something'
    assert result['ranking'][0] == result['family_nodes'][0], (
        'stale node %r should head the ranking; ranking=%r'
        % (result['family_nodes'][0], result['ranking']))
    assert result['current_node'] == result['family_nodes'][-1]
    assert result['current_node'] != result['ranking'][0], (
        'the current value must NOT rank first under the feed-only trap')

    # Every stale value is still readable from an active node -- this is
    # exactly what let the reader answer "17 credits" in LT1.
    for stale in [t['value'] for t in CORRECTION_TURNS[:-1]]:
        assert stale in result['active_text'], (
            'stale value %r must still be reachable under feed-only' % stale)


def test_supersede_retires_old_value_and_ranks_new_first(tmp_path):
    """The fix: the production correction path retires the old value."""
    result = supersede_arm(tmp_path / 'supersede')

    stale_values = [t['value'] for t in CORRECTION_TURNS[:-1]]
    current_value = CORRECTION_TURNS[-1]['value']

    # Old values are retired and inactive.
    assert result['retired_nodes'], 'supersession must retire the old nodes'
    for node in result['retired_nodes']:
        assert result['nodes'][node]['retired'] is True
        assert result['nodes'][node]['active'] is False
        assert result['nodes'][node]['superseded_by'], (
            'a retired node must name its successor')

    # Retired nodes are gone from the eligible route base -- they can never be
    # ranked, let alone mounted. This is the assertion that makes the 5 LT1
    # `recall_4_*` rows impossible once the fixture is fixed.
    for node in result['retired_nodes']:
        assert node not in result['route_cand_base'], (
            'retired node %d must not be route-eligible' % node)

    # The current value survives as an active, eligible candidate and ranks
    # first for the probe.
    assert result['current_node'] in result['route_cand_base']
    assert result['ranking'], 'the current value must still be routable'
    assert result['ranking'][0] == result['current_node'], (
        'the current value must rank first; ranking=%r' % result['ranking'])

    # No stale text is reachable from any active node.
    for stale in stale_values:
        assert stale not in result['active_text'], (
            'stale value %r must not survive in any active node' % stale)
    assert current_value in result['active_text']


def test_the_supersede_assertions_actually_fail_on_the_old_lineage(tmp_path):
    """RED proof: the fix's assertions must be capable of failing.

    A gate that cannot go RED proves nothing. Run the three load-bearing
    supersede assertions against the UNFIXED (feed-only) lineage and require
    each one to fail there.
    """
    feed = feed_only_arm(tmp_path / 'red')
    failures = []

    # 1. "the old nodes are retired"
    try:
        assert feed['retired_nodes'], 'supersession must retire the old nodes'
    except AssertionError:
        failures.append('retired_nodes')

    # 2. "retired nodes leave the route candidate base"
    try:
        for node in feed['family_nodes'][:-1]:
            assert node not in feed['route_cand_base']
    except AssertionError:
        failures.append('route_cand_base')

    # 3. "the current value ranks first"
    try:
        assert feed['ranking'][0] == feed['current_node']
    except AssertionError:
        failures.append('ranking_first')

    assert failures == ['retired_nodes', 'route_cand_base', 'ranking_first'], (
        'all three supersede assertions must go RED on the old lineage; '
        'only these failed: %r' % failures)


def test_the_two_arms_differ_only_in_lineage(tmp_path):
    """Both arms see the same three turns; only the edges differ."""
    feed = feed_only_arm(tmp_path / 'a')
    sup = supersede_arm(tmp_path / 'b')
    assert feed['turns_applied'] == sup['turns_applied'] == len(CORRECTION_TURNS)
    assert feed['retired_nodes'] == []
    assert sup['retired_nodes'] != []
    # The only difference that matters to routing: candidate-base size.
    assert len(sup['route_cand_base']) < len(feed['route_cand_base']) or (
        set(sup['retired_nodes']).isdisjoint(sup['route_cand_base']))
