#!/usr/bin/env python3
"""GRM-D1: per-row cause table for the LT1 arm-A wrong rows.

Reads the frozen LT1 amendment-2 margin_first receipts (read-only) and the
frozen fixture, joins each probe row to its route receipt, and classifies the
cause. No model is run; every field is copied from a receipt on disk.

Prior art:
  * The classification axes (fixture-lineage vs ranking vs composition) are
    RD2's C7 finding (GRM contributors, 2026) restated for LT1; taken: the
    two named traps (correction registered as an ordinary fact; alias edge
    mounted without its base). Mine: the per-row join against
    `_route_observation` and the node-id -> deposit-turn reconstruction.
  * Node-id reconstruction mirrors scripts/grm_lt1_worker.py's own
    `state['turn_nodes']` assignment order (GRM contributors, 2026), recomputed
    here from the fixture so the table needs no checkpoint.
  * No prior art known to me for this exact table shape.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_lt1 import score  # noqa: E402

# GRM-F6: LT1 was Path('/mnt/ForgeRealm/wt/grm-lt1') -- a pruned seat
# worktree. Here the dead path did not pass vacuously, it SKIPPED: the test
# module carries `skipif(not ct.CELLS.exists())`, so 12 gates were quietly
# deselected instead of run. The amendment-2 margin_first cells and the LT1
# fixture are both canonical in THIS repo, so the rebind makes them live.
# Prior art: repo-root-from-__file__ (setuptools / pytest rootdir, 2009-).
# See scripts/grm_repo_paths.py.
from scripts import grm_repo_paths as repo_paths  # noqa: E402

LT1 = repo_paths.repo_root()
CELLS = repo_paths.receipt_root(
    'artifacts/grm_lt1/amendment2/run_margin_first/cells', 'GRM_LT1_CELLS_ROOT',
    "scripts/grm_d1_recap.py's `receipt=` field (absolute .../grm-lt1/... path)",
    '/mnt/ForgeRealm/wt/grm-lt1/artifacts/grm_lt1/amendment2/run_margin_first/cells')
FIXTURE = LT1 / 'fixtures/lt1/dialogue.json'
OUT = ROOT / 'artifacts/grm_d1'

# Cause classes named by the GRM-D1 order.
FIXTURE_LINEAGE = 'fixture-lineage (harness)'
# The order names this class "stale-first ranking (core A-DEC)". The receipts
# show the sharper truth for the one arm-A row that lands here: A-DEC ranked
# the CORRECT node first and made it the whole rank plan; the probe LADDER then
# discarded that mount as ungrounded and served a later trip's unrelated node.
# The label keeps the order's name so the lead can map classes 1:1, and adds the
# ladder qualifier so the ruling is not overclaimed against A-DEC's ranking.
STALE_FIRST = ('stale-first ranking (core A-DEC) '
               '[arm-A instance is ladder-drop, not mis-rank]')
ALIAS_NO_BASE = 'alias-edge-without-base (core composition)'
READER_WRONG = 'reader wrong value with the right mount'
OTHER = 'other'


def node_turn_map(fixture):
    """Node id -> depositing turn, exactly as the worker assigns them."""
    out, nid = {}, 0
    for turn in fixture['turns']:
        if turn['kind'] not in ('probe', 'recap'):
            out[nid] = turn
            nid += 1
    return out


def load_rows(arm):
    rows = []
    # GRM-F6 vacuous-zero guard: 'no wrong rows' must not mean 'no rows'.
    for path in repo_paths.require_rows(sorted(glob.glob(str(CELLS / f'{arm}-*/probes.jsonl'))), CELLS, f'{arm}-*/probes.jsonl', f'D1 LT1 arm-{arm} cell census'):
        ctl = Path(path).parent / 'controller.json'
        if not ctl.exists() or json.loads(ctl.read_text()).get('status') != 'COMPLETE':
            continue
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row['_receipt'] = str(path)
                rows.append(row)
    return rows


def classify(probe, row, n2t):
    """Return (class, because) for one wrong row, from receipt fields only."""
    ri = row['memory']['route_info']
    seated = list(ri.get('fit_seated') or [])
    ranking = list(ri.get('ranking_ids') or [])
    cls = probe['class']
    served_turns = [n2t[i]['turn'] for i in seated if i in n2t]
    # The node holding the currently-correct value, by fixture lineage.
    family = sorted(i for i, t in n2t.items()
                    if t.get('entity') == probe['entity']
                    and t.get('attribute') == probe['attribute'])
    current_node = max(family) if family else None

    if cls == 'correction':
        # The whole correction family is still an ACTIVE route candidate:
        # nothing was ever retired. That is visibly upstream of the ranking --
        # core's `_route_cand_base` (core/graft_arena.py:1450-1454) drops
        # retired nodes, so a real supersession would have removed the stale
        # ones from `ranking_ids` entirely.
        if all(i in ranking for i in family):
            if current_node is not None and seated and max(seated) < current_node:
                return FIXTURE_LINEAGE, (
                    'every node of the correction family %s (turns %s) is still an '
                    'active route candidate (ranking_ids=%s): no supersession edge '
                    'was ever written by the harness, so stale node %d (turn %d, '
                    'value %r) outranks and is the only node seated, while the '
                    'current node %d (turn %d) is ranked last and never mounted'
                    % (family, [n2t[i]['turn'] for i in family], ranking,
                       seated[0], n2t[seated[0]]['turn'], n2t[seated[0]].get('value'),
                       current_node, n2t[current_node]['turn']))
            return FIXTURE_LINEAGE, (
                'correction family %s all still active in ranking_ids=%s; no '
                'supersession edge exists' % (family, ranking))
        return OTHER, 'correction family not fully present in ranking'

    if cls == 'alias':
        alias_nodes = [i for i, t in n2t.items()
                       if t.get('kind') == 'alias' and t.get('entity') == probe['entity']]
        identified = list(ri.get('admission_identified_candidates') or [])
        profile = ri.get('_route_observation', {}).get('admission_profile', {})
        # The alias EDGE is what the identifier matched; the BASE fact is what
        # got seated. Nothing composes the two, so the mount carries the base
        # text with no binding from the queried alias name to it.
        if any(i in identified for i in alias_nodes) and not (set(alias_nodes) & set(seated)):
            return ALIAS_NO_BASE, (
                'the identifier token(s) %s matched the alias edge node(s) %s '
                '(admission_identified_candidates=%s), but the rank plan and mount '
                'seated only the base fact node(s) %s (turns %s); core has no '
                'alias->base composition (no alias handling exists in '
                'core/graft_arena.py or core/graft_repository.py), so the alias '
                'edge is never mounted together with its base and the reader sees '
                'a base fact it cannot bind to the queried name'
                % (profile.get('identifier_tokens'), alias_nodes, identified,
                   seated, served_turns))
        return OTHER, 'alias row without the identified-edge/seated-base split'

    # fresh
    if current_node is not None and current_node in ranking and current_node not in seated:
        return STALE_FIRST, (
            'the correct node %d (turn %d) is rank-1 in ranking_ids=%s and is the '
            'entire admission_rank_plan=%s, but fit_seated=%s (nodes at turns %s): '
            'the ladder\'s first trip mounted the plan head and came back '
            'ungrounded, and a LATER trip seated an unrelated same-attribute node '
            '(served_without_plan_head=%s). Routing picked the right node; the '
            'ladder discarded it.'
            % (current_node, n2t[current_node]['turn'], ranking,
               ri.get('admission_rank_plan'), seated, served_turns,
               ri.get('served_without_plan_head')))
    if current_node is not None and current_node in seated:
        return READER_WRONG, (
            'the correct node %d (turn %d) WAS seated (fit_seated=%s) and the '
            'reader still produced a wrong value'
            % (current_node, n2t[current_node]['turn'], seated))
    return OTHER, 'fresh row with neither pattern'


def build(arm='A'):
    fixture = json.loads(FIXTURE.read_text())
    probes = {p['id']: p for p in fixture['probes']}
    n2t = node_turn_map(fixture)
    rows = load_rows(arm)
    table, counts = [], {}
    for row in rows:
        probe = probes[row['probe_id']]
        graded = score(row['memory']['answer'], probe['expected'])
        if graded['category'] == 'correct':
            continue
        ri = row['memory']['route_info']
        obs = ri.get('_route_observation', {})
        cause, because = classify(probe, row, n2t)
        counts[cause] = counts.get(cause, 0) + 1
        seated = list(ri.get('fit_seated') or [])
        ranked = list(ri.get('ranking_ids') or [])
        family = sorted(i for i, t in n2t.items()
                        if t.get('entity') == probe['entity']
                        and t.get('attribute') == probe['attribute'])
        alias_nodes = [i for i, t in n2t.items()
                       if t.get('kind') == 'alias' and t.get('entity') == probe['entity']]
        current_node = max(family) if family else None
        table.append(dict(
            question_id=row['probe_id'],
            probe_turn=row['turn'],
            distance=probe['distance'],
            probe_class=probe['class'],
            question=probe['question'],
            expected=probe['expected'],
            served_value=row['memory']['answer'],
            score_category=graded['category'],
            ranked_nodes=[dict(node=i, deposited_turn=n2t[i]['turn'],
                               kind=n2t[i]['kind'], value=n2t[i].get('value'))
                          for i in ranked if i in n2t],
            mounted_nodes=[dict(node=i, deposited_turn=n2t[i]['turn'],
                                kind=n2t[i]['kind'], value=n2t[i].get('value'))
                           for i in seated if i in n2t],
            recency_mounted_ids=ri.get('recency_mounted_ids'),
            rs3_residency=row['memory'].get('residency'),
            current_value_node=current_node,
            current_value_node_turn=(n2t[current_node]['turn']
                                     if current_node is not None else None),
            current_value_node_exists=current_node is not None,
            # Nothing in LT1 is ever retired: the worker feeds every turn with
            # a plain arena.feed() (scripts/grm_lt1_worker.py:147).
            current_value_node_superseded=False,
            current_value_node_mounted=(current_node in seated
                                        if current_node is not None else None),
            alias_edge_nodes=alias_nodes,
            alias_edge_mounted=bool(set(alias_nodes) & set(seated)),
            base_fact_mounted=bool(set(family) & set(seated)) if family else None,
            admission_rule=ri.get('admission_rule'),
            admission_policy=ri.get('admission_policy'),
            admission_policy_branch=ri.get('admission_policy_branch'),
            admission_identified_candidates=ri.get('admission_identified_candidates'),
            admission_rank_plan=ri.get('admission_rank_plan'),
            admission_route_margin_1_2=ri.get('admission_route_margin_1_2'),
            admission_margin_threshold=ri.get('admission_margin_threshold'),
            served_from=obs.get('serving_path'),
            served_without_plan_head=ri.get('served_without_plan_head'),
            rt1_provenance=dict(
                route_backend=obs.get('admission_profile', {}).get('route_backend'),
                rule_sha256=obs.get('admission_profile', {}).get('rule_sha256'),
                identifier_tokens=obs.get('admission_profile', {}).get('identifier_tokens'),
                rare_identifier_tokens=obs.get('admission_profile', {}).get('rare_identifier_tokens'),
                excluded_live_ids=obs.get('excluded_live_ids'),
                route_limit=obs.get('route_limit'),
                trips=obs.get('trips'),
            ),
            cause_class=cause,
            because=because,
            receipt=row['_receipt'],
        ))
    return dict(arm=arm, wrong_rows=len(table), class_counts=counts, rows=table)


def markdown(data):
    out = ['# GRM-D1 cause table — LT1 arm %s (%d wrong rows)'
           % (data['arm'], data['wrong_rows']), '',
           '| # | question id | dist | class | expected | served value | ranked (node@turn) | seated (node@turn) | current-value node | superseded? | mounted? | alias edge mounted / base mounted | admission_rule / branch | served_from | cause class | receipt |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for n, r in enumerate(data['rows'], 1):
        rk = ' '.join('%d@%d' % (x['node'], x['deposited_turn']) for x in r['ranked_nodes'])
        mt = ' '.join('%d@%d' % (x['node'], x['deposited_turn']) for x in r['mounted_nodes'])
        out.append('| %d | `%s` | %d | %s | `%s` | `%s` | %s | %s | %s@%s | %s | %s | %s / %s | %s / %s | `%s` | **%s** | `%s` |' % (
            n, r['question_id'], r['distance'], r['probe_class'], r['expected'],
            r['served_value'].replace('|', '\\|')[:70], rk, mt,
            r['current_value_node'], r['current_value_node_turn'],
            'no' if not r['current_value_node_superseded'] else 'yes',
            'yes' if r['current_value_node_mounted'] else 'no',
            'yes' if r['alias_edge_mounted'] else 'no',
            'yes' if r['base_fact_mounted'] else 'no',
            r['admission_rule'], r['admission_policy_branch'],
            r['served_from'], r['cause_class'], r['receipt']))
    out += ['', '## Class counts', '']
    for k, v in sorted(data['class_counts'].items(), key=lambda kv: -kv[1]):
        out.append('- **%s**: %d' % (k, v))
    out += ['', '## Per-row reasoning (receipt fields only)', '']
    for n, r in enumerate(data['rows'], 1):
        out += ['### %d. `%s` (d=%d, %s) — %s'
                % (n, r['question_id'], r['distance'], r['probe_class'], r['cause_class']),
                '', '- question: %s' % r['question'],
                '- expected: `%s`' % r['expected'],
                '- served: `%s`' % r['served_value'],
                '- ranked: %s' % ', '.join('node %d = turn %d (%s)'
                                           % (x['node'], x['deposited_turn'], x['kind'])
                                           for x in r['ranked_nodes']),
                '- seated: %s' % ', '.join('node %d = turn %d (%s, value=%r)'
                                           % (x['node'], x['deposited_turn'], x['kind'], x['value'])
                                           for x in r['mounted_nodes']),
                '- RS3 residency: %s' % json.dumps(r['rs3_residency']),
                '- admission: rule=%s policy=%s branch=%s margin_1_2=%s threshold=%s'
                % (r['admission_rule'], r['admission_policy'], r['admission_policy_branch'],
                   r['admission_route_margin_1_2'], r['admission_margin_threshold']),
                '- served_from: `%s` (served_without_plan_head=%s)'
                % (r['served_from'], r['served_without_plan_head']),
                '- RT1 provenance: backend=%s rule_sha256=%s identifier_tokens=%s'
                % (r['rt1_provenance']['route_backend'],
                   r['rt1_provenance']['rule_sha256'],
                   r['rt1_provenance']['identifier_tokens']),
                '- because: %s' % r['because'],
                '- receipt: `%s`' % r['receipt'], '']
    return '\n'.join(out) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ('A', 'B'):
        data = build(arm)
        (OUT / ('cause_table_%s.json' % arm)).write_text(
            json.dumps(data, indent=2, sort_keys=True) + '\n')
        (OUT / ('cause_table_%s.md' % arm)).write_text(markdown(data))
        print(arm, data['wrong_rows'], data['class_counts'])


if __name__ == '__main__':
    main()
