#!/usr/bin/env python3
"""GRM-D1: the two LT1 correction arms, on CPU, with no model quality claim.

`feed_only_arm` reproduces the LT1 harness lineage exactly: every turn -- fact
AND correction -- is deposited with a bare `arena.feed()` (the shape at
scripts/grm_lt1_worker.py:147, whose own comment reads "User corrections are
ordinary prose, not hidden supersede calls"). Nothing is retired, so the whole
correction family stays route-eligible and the stale node can outrank and
outseat the current one. That is the trap behind LT1's five wrong
`recall_4_*` rows.

`supersede_arm` applies the SAME three turns through the production correction
path -- `repo.apply_memory_command("correct memory: <old> => <new>")`, which is
what `scripts/grm_e2e_session.py:2590-2597` runs for `kind == "supersede"`.
`core/graft_repository.py:correct_memory` then retires the matched active
nodes (`retired=True`, `metadata.active=False`, `metadata.superseded_by=[new]`)
and bumps the route epoch, so `core/graft_arena.py:_route_cand_base` drops them
from the candidate base entirely.

No core file is modified. Both arms read the same production code.

Prior art:
  * The `correct memory: <old> => <new>` command string and the
    `kind='supersede'` + `correction_command` fixture shape come from C7 r3
    (`wt/grm-c7` scripts/grm_c7_register_r3.py:41-49, GRM contributors 2026).
    Taken verbatim.
  * `correct_memory`, `_route_cand_base`, `route`, and the C7 CPU doubles
    (scripts/grm_c7_diagnose.py) are existing GRM code, used unchanged.
  * Mine: the paired feed-only contrast arm, and asserting the fix at the
    ROUTE-ELIGIBILITY layer (`_route_cand_base`) rather than at the answer
    layer, so the proof does not depend on any reader.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The Vega docking-fee lineage, copied verbatim out of the frozen LT1 fixture
# (fixtures/lt1/dialogue.json turns 1, 3, 15) -- the family behind the five
# wrong `recall_4_*` rows.
CORRECTION_TURNS = [
    dict(turn=1, kind='fact', entity='Vega', attribute='docking fee',
         value='17 credits',
         user="Vega's docking fee will be 17 credits.",
         assistant="All right, I'll keep that in mind as we work through the layout."),
    dict(turn=3, kind='correction', entity='Vega', attribute='docking fee',
         value='23 credits', previous_turn=1,
         user="Actually, Vega's docking fee will be 23 credits, replacing the earlier choice.",
         assistant="Understood; I'll use the revised choice."),
    dict(turn=15, kind='correction', entity='Vega', attribute='docking fee',
         value='29 credits', previous_turn=3,
         user="Actually, Vega's docking fee will be 29 credits, replacing the earlier choice.",
         assistant="Understood; I'll use the revised choice."),
]

PROBE = "What did we settle on for Vega's docking fee?"


def correction_command(turn, previous_value):
    """The C7 r3 command shape, filled in from an LT1 correction turn.

    Prior art: scripts/grm_c7_register_r3.py:48-49 (GRM contributors, 2026),
    verbatim `correct memory: <old> => <new>` grammar; the query half names the
    value being replaced so `correct_memory` matches exactly the stale node(s).
    """
    return 'correct memory: %s => %s' % (previous_value, turn['user'])


def _open(path):
    import pytest
    from scripts.grm_c7_diagnose import repository
    patch = pytest.MonkeyPatch()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    repo = repository(path / 'repository', patch)
    repo.arena.recency_mounts = 2
    return repo, patch


def _snapshot(repo):
    """Read the lineage facts straight off the arena. No inference."""
    arena = repo.arena
    nodes = {}
    for i, g in enumerate(arena.grafts):
        meta = g.get('metadata') or {}
        nodes[i] = dict(
            text=g.get('text', ''),
            kind=g.get('kind', 'turn'),
            retired=bool(g.get('retired')),
            active=bool(meta.get('active', True)),
            superseded_by=list(meta.get('superseded_by') or []),
            supersedes=list(meta.get('supersedes') or []),
        )
    base = [int(i) for i in arena._route_cand_base()]
    retired = sorted(i for i, n in nodes.items() if n['retired'])
    ranked = arena.route(PROBE, exclude=set(), limit=6, route_backend='python')
    ranking = []
    for item in (ranked or []):
        ranking.append(int(item[0]) if isinstance(item, (list, tuple)) else int(item))
    active_text = ' '.join(nodes[i]['text'] for i in base if i in nodes)
    return dict(nodes=nodes, route_cand_base=base, retired_nodes=retired,
                ranking=ranking, active_text=active_text)


def _family_nodes(snapshot):
    """Node ids whose text carries this lineage's entity+attribute."""
    return sorted(i for i, n in snapshot['nodes'].items()
                  if 'Vega' in n['text'] and 'docking fee' in n['text'])


def feed_only_arm(path):
    """LT1's actual lineage: every turn is a bare feed(); nothing retires."""
    from scripts import grm_e2e_session as e2e
    repo, patch = _open(path)
    try:
        arena = repo.arena
        for turn in CORRECTION_TURNS:
            # Prior art: scripts/grm_lt1_worker.py:147 (GRM contributors,
            # 2026), reproduced verbatim -- this IS the trap under test.
            idx = arena.feed(e2e.harmony_turn(turn['user'], turn['assistant']))
            arena.grafts[idx]['kind'] = 'turn'
        snap = _snapshot(repo)
    finally:
        patch.undo()
        repo.close()
    family = _family_nodes(snap)
    snap.update(arm='feed_only', turns_applied=len(CORRECTION_TURNS),
                family_nodes=family,
                current_node=family[-1] if family else None)
    return snap


def supersede_arm(path):
    """The C7 r3 shape: corrections go through the production supersede path."""
    from scripts import grm_e2e_session as e2e
    repo, patch = _open(path)
    try:
        arena = repo.arena
        previous_value = None
        for turn in CORRECTION_TURNS:
            if turn['kind'] == 'correction':
                # Prior art: scripts/grm_e2e_session.py:2590-2597 (GRM, 2026).
                # The production supersede turn: the authoritative correction
                # command retires the stale node and writes the replacement.
                repo.apply_memory_command(correction_command(turn, previous_value))
            else:
                idx = arena.feed(e2e.harmony_turn(turn['user'], turn['assistant']))
                arena.grafts[idx]['kind'] = 'turn'
            previous_value = turn['value']
        snap = _snapshot(repo)
    finally:
        patch.undo()
        repo.close()
    active_family = sorted(i for i in _family_nodes(snap)
                           if not snap['nodes'][i]['retired'])
    snap.update(arm='supersede', turns_applied=len(CORRECTION_TURNS),
                family_nodes=_family_nodes(snap),
                current_node=active_family[-1] if active_family else None)
    return snap


def main():
    out = ROOT / 'artifacts/grm_d1'
    out.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        feed = feed_only_arm(Path(tmp) / 'feed_only')
        sup = supersede_arm(Path(tmp) / 'supersede')
    receipt = dict(
        evidence_class='CPU lineage fixture (no model quality claim)',
        probe=PROBE,
        correction_turns=CORRECTION_TURNS,
        feed_only=dict(retired_nodes=feed['retired_nodes'],
                       route_cand_base=feed['route_cand_base'],
                       ranking=feed['ranking'],
                       family_nodes=feed['family_nodes'],
                       current_node=feed['current_node'],
                       stale_still_reachable=[t['value'] for t in CORRECTION_TURNS[:-1]
                                              if t['value'] in feed['active_text']]),
        supersede=dict(retired_nodes=sup['retired_nodes'],
                       route_cand_base=sup['route_cand_base'],
                       ranking=sup['ranking'],
                       family_nodes=sup['family_nodes'],
                       current_node=sup['current_node'],
                       stale_still_reachable=[t['value'] for t in CORRECTION_TURNS[:-1]
                                              if t['value'] in sup['active_text']],
                       superseded_by={str(i): sup['nodes'][i]['superseded_by']
                                      for i in sup['retired_nodes']}),
    )
    (out / 'supersession_cpu_receipt.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
