"""GRM-F1 follow-up 4 — the r3 result, read back from the committed receipts.

These tests assert facts about a campaign that ALREADY RAN. They exist so the
claims in `artifacts/grm_f1/REPORT.md` §r3 and amendment 13 cannot drift from
the receipts they were read out of: every number here is recomputed from
`artifacts/grm_f1/lt1_1_r3/run_*` rather than restated.

The headline result they pin: **F1+F2 without A1 is an alias REGRESSION**
(A' 7/10 against the control's 10/10), caused by two individually-correct
treatments interacting — F2 makes alias turns unfoldable, so they are never
superseded and survive as lone raw edges; F1 inflates the rank plan; the lone
edge is then evicted for width.

Prior art: the receipt-reading shape is `scripts/grm_a1_report.py` and the D1
cause-table work (GRM contributors, 2026), reused. New here: the per-row
eviction attribution and the fold-source-list comparison that establishes
F2's exclusion without `fold_guard_history`. No prior art known to me for
this exact suite. CPU author evidence only: this seat ran no GPU work; the
campaign was run by the lead.
"""
import glob
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / 'artifacts/grm_f1/lt1_1_r3'
ARMS = ('A0', 'Aprime', 'Aplusprime')

pytestmark = pytest.mark.skipif(
    not (R3 / 'run_A0/cells').is_dir(),
    reason='r3 receipts absent (campaign artifacts are gitignored)')


def probes(arm):
    out = {}
    for f in sorted(glob.glob(str(R3 / f'run_{arm}/cells/*/probes.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            out[r['probe_id']] = r
    return out


def manifest(arm):
    cells = sorted(glob.glob(
        str(R3 / f'run_{arm}/cells/*/checkpoint/repository/manifest.json')))
    return json.loads(Path(cells[-1]).read_text())['nodes']


def summary(arm):
    return json.loads((R3 / f'lead_{arm}_summary.json').read_text())


# ------------------------------------------------------------- the scores

@pytest.mark.parametrize('arm,expected', [
    ('A0', dict(fresh='10/15', correction='9/10', alias='10/10', recap='4/5')),
    ('Aprime', dict(fresh='11/15', correction='9/10', alias='7/10',
                    recap='5/5')),
    ('Aplusprime', dict(fresh='13/15', correction='10/10', alias='10/10',
                        recap='4/5')),
])
def test_measured_scores(arm, expected):
    s = summary(arm)
    assert s['complete'] is True
    assert s['complete_cells'] == s['total_cells'] == 26
    got = {k: '%d/%d' % (v['correct'], v['expected_n'])
           for k, v in s['by_class'].items()}
    assert got == expected, got


def test_a0_reproduces_r2_arm_a_so_attribution_is_licensed():
    """The control arm's whole job (registration `attribution_rule`)."""
    r2 = json.loads(
        (ROOT / 'artifacts/grm_d1/lt1_1/lead_A_r2_summary.json').read_text())
    a0 = summary('A0')
    for cls in ('fresh', 'correction', 'alias', 'recap'):
        assert (a0['by_class'][cls]['correct']
                == r2['by_class'][cls]['correct']), cls


def test_a_prime_is_a_net_regression_on_the_verifiable_column():
    """The c2 column is not in any r3 receipt; on the column that IS, the
    F1+F2 arm is a net LOSS against the control."""
    totals = {}
    for arm in ARMS:
        rows = probes(arm)
        totals[arm] = sum(
            1 for r in rows.values() if r['memory']['score']['exact_correct'])
    assert totals == {'A0': 33, 'Aprime': 32, 'Aplusprime': 37}, totals


def test_column_two_is_absent_from_every_receipt():
    """Pinned so no future reader assumes c2 can be audited from r3."""
    for arm in ARMS:
        for r in probes(arm).values():
            assert not any(k.startswith('col2') for k in r['memory']['score'])
        s = summary(arm)
        assert set(s['by_class']['fresh']) == {
            'correct', 'n', 'expected_n', 'exact_rate'}


def test_alias_column_regressed_in_the_f1_f2_arm():
    """THE finding: F1+F2 without A1 is a regression, not a partial win."""
    assert summary('Aprime')['by_class']['alias']['correct'] == 7
    assert summary('A0')['by_class']['alias']['correct'] == 10
    assert summary('Aplusprime')['by_class']['alias']['correct'] == 10


# ------------------------------------------------------ the differing rows

def test_eleven_rows_differ_not_nine():
    data = {a: probes(a) for a in ARMS}
    ids = sorted(set().union(*[set(d) for d in data.values()]))
    differ = [p for p in ids
              if len({bool(data[a][p]['memory']['score']['exact_correct'])
                      for a in ARMS if p in data[a]}) > 1]
    assert len(differ) == 11, sorted(differ)
    assert 'recall_1_25' in differ and 'recall_1_50' in differ


@pytest.mark.parametrize('row', ['recall_3_10', 'recall_5_50', 'recall_7_10',
                                 'recall_7_25', 'recall_7_50'])
def test_every_a_prime_loss_is_a_width_eviction(row):
    """ONE mechanism, named by the route receipt on all five losses."""
    r = probes('Aprime')[row]
    info = r['memory']['route_info']
    assert r['memory']['score']['exact_correct'] is False
    assert info['mount_dropped_for_width'], info.get('mount_plan')
    # The plan grew past what the arena can seat.
    nodes = manifest('Aprime')
    planned = sum(int(nodes[i]['ntok']) for i in info['mount_plan'])
    assert planned > r['memory']['residency']['width'], (planned, row)
    assert len(info['mount_plan']) == 3, info['mount_plan']


def test_recall_7_10_drops_the_only_node_holding_the_alias_relation():
    """The alias failure, at node resolution."""
    r = probes('Aprime')['recall_7_10']
    info = r['memory']['route_info']
    assert info['mount_plan'] == [12, 5, 18]
    assert info['fit_seated'] == [5, 12]
    assert info['mount_dropped_for_width'] == [18]
    # Admission HAD identified it -- this is eviction, not a ranking miss.
    assert info['admission_identified_candidates'] == [18]
    nodes = manifest('Aprime')
    assert 'the Beacon' in nodes[18]['text']
    # And nothing seated carries the alias relation.
    for i in info['fit_seated']:
        assert 'Beacon' not in nodes[i]['text']
    assert 'still working on that' in r['memory']['answer']


def test_recall_5_50_twelve_is_a_cross_entity_leak():
    """The served "12" is Breakwater's z-coordinate in a co-mounted node."""
    r = probes('Aprime')['recall_5_50']
    nodes = manifest('Aprime')
    seated = r['memory']['route_info']['fit_seated']
    assert '12' in r['memory']['answer']
    assert not any('Medibay' in nodes[i]['text'] for i in seated)
    leak = [i for i in seated if '(-31, 48, 12)' in nodes[i]['text']]
    assert leak, seated


def test_a_plus_prime_merged_digest_serves_the_alias():
    """A1 removes exactly the failure: relation AND value in ONE node."""
    r = probes('Aplusprime')['recall_7_10']
    info = r['memory']['route_info']
    assert r['memory']['score']['exact_correct'] is True
    assert info['mount_plan'] == [20]
    assert not info['mount_dropped_for_width']
    node = manifest('Aplusprime')[20]
    assert 'Beacon' in node['text'] and '18 October 2196' in node['text']
    assert node['metadata']['alias_merge']['lineage']
    assert node['metadata']['supersedes'] == [19]


# ------------------------------------------- the Beacon capture (item 3)

@pytest.mark.parametrize('arm,count,capture', [
    ('A0', 3, True), ('Aprime', 1, False), ('Aplusprime', 2, False)])
def test_beacon_nodes_per_arm(arm, count, capture):
    nodes = manifest(arm)
    hits = [g for g in nodes if 'Beacon' in (g.get('text') or '')]
    assert len(hits) == count, [n.get('kind') for n in hits]
    # The capture: another entity's fact re-filed onto the alias.
    captured = [g for g in hits
                if 'crew of the Beacon' in g['text']
                or 'position of the Beacon' in g['text']]
    assert bool(captured) is capture, [g.get('kind') for g in captured]


def test_f2_excluded_the_alias_turns_from_the_fold_window():
    """Same fold, same cell: the SOURCE LIST is the receipt.

    `fold_guard_history` is not persisted into the cell receipts, so F2's
    exclusion is established by comparing what each arm's fold consumed.
    """
    a0 = manifest('A0')
    ap = manifest('Aprime')
    assert a0[24]['sources'] == [13, 14, 17, 18]
    assert 'the Beacon' in a0[24]['text']
    assert ap[24]['sources'] == [13, 14]
    assert 'Beacon' not in ap[24]['text']
    assert 'Commtower' in ap[24]['text'] and 'Breakwater' in ap[24]['text']


def test_the_a_prime_alias_edge_is_the_source_of_nothing():
    """Never folded therefore never superseded: a lone raw edge forever."""
    nodes = manifest('Aprime')
    edge = nodes[18]
    assert not edge.get('retired')
    assert edge['metadata']['active'] is True
    assert edge['metadata'].get('digest_of') is None
    assert not any(18 in (g.get('sources') or []) for g in nodes)


# ------------------------------------------------------------- residency

@pytest.mark.parametrize('arm,nodes_n,active_n,max_seats', [
    ('A0', 199, 36, 94), ('Aprime', 228, 213, 95),
    ('Aplusprime', 200, 175, 94)])
def test_residency_bounded_in_every_arm(arm, nodes_n, active_n, max_seats):
    nodes = manifest(arm)
    assert len(nodes) == nodes_n
    assert sum(1 for g in nodes if not g.get('retired')) == active_n
    seen = 0
    for f in glob.glob(str(R3 / f'run_{arm}/cells/*/residency.jsonl')):
        for line in open(f):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seen = max(seen, row.get('summed_token_seats', 0))
            assert row['summed_token_seats'] <= (
                2 * row['width'] + row['actual_recency_token_seats'])
    assert seen == max_seats


# ------------------------------------------------------------ amendment 13

def test_amendment13_binds_the_receipts_and_the_finding():
    amd = json.loads(
        (ROOT / 'artifacts/grm_d1/lt1_1/amendment13.json').read_text())
    assert amd['amendment'] == 13
    assert amd['gpu_executed'] is True
    assert amd['differing_row_count'] == 11
    assert amd['finding']['id'] == 'F1-R3-1'
    assert 'DO NOT merge F1+F2 without A1' in amd['finding']['merge_guidance']
    import hashlib
    for label, rec in amd['arm_receipts'].items():
        assert rec['cells_complete'] == 26, label
        path = ROOT / rec['summary_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            rec['summary_sha256']), label
    # The alias falsifier is recorded as FAILED, not softened.
    assert 'FAILED' in amd['prediction_verdicts']["A'"]['aliases']['verdict']
    # And the open flake is carried as RED, not quietly dropped.
    # The cause table is bound by sha, so the tables in the REPORT cannot
    # drift from the receipts they were computed out of.
    # The c2 column is recorded as UNVERIFIABLE, not silently carried.
    cols = amd['scorer_columns']
    assert cols['primary']['verifiable_from_receipts'] is True
    assert cols['column_2']['verifiable_from_receipts'] is False
    assert cols['primary']['totals'] == {
        'A0': '33/40', "A'": '32/40', "A+'": '37/40'}
    ct = amd['cause_table']
    assert ct['rows'] == 11
    assert hashlib.sha256(
        (ROOT / ct['path']).read_bytes()).hexdigest() == ct['sha256']
    red = amd['gate_defect_found']
    assert red['id'] == 'F1-R3-RED1'
    assert red['status'] == 'ROOT-CAUSED AND FIXED'
    assert 'GptOss20B_TC' in red['root_cause']
    assert red['method_note'] and red['successor']
    blob = json.dumps(amd)
    assert '/mnt/' not in blob and 'wt/' not in blob
