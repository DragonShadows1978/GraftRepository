#!/usr/bin/env python3
"""GRM-F1 follow-up 4: amendment 13 -- BIND the r3 receipts and the finding.

WHAT THIS AMENDMENT IS FOR.  Amendments 10-12 registered r3 BEFORE it ran.
r3 has now run to completion on the card, three arms, 26/26 cells each.
Amendment 13 closes the loop: it binds the receipt directories and summaries
by sha, records the MEASURED result against each registered prediction, and
registers the finding the campaign produced -- that F1+F2 WITHOUT A1 is an
alias regression, not a partial improvement.

NO CORE OR RUNNER CHANGE.  Follow-up 4 is analysis only.  Amendment 12's
core_rebind and runner binding are carried forward verbatim and re-verified
against the tree; this script RAISES if any of them moved.

THE FINDING, registered so a later merge decision cannot be taken without it:
the four flags are a SET at this arena width.  A' (F1+F2+F5, A1 OFF) scored
aliases 7/10 against the control's 10/10.  Cause, from
`route_info.mount_dropped_for_width` on all five A' losses: F2 makes alias
turns unfoldable (correctly -- it kills the capture defect present in A0),
so they are never superseded either and survive as lone raw edges; F1
inflates the rank plan from one member to three; the lone edge is admitted,
ranked third, and evicted for width (plan 154 seats vs width 96).  A1's
write-time merge puts relation and value in ONE node and removes exactly
that failure.

PRIOR ART.
  * `scripts/grm_f1_register_lt11_r3_amd12.py` (this seat, 2026-09-11) --
    the amendment structure, the carried-forward re-verification and the
    measured-not-typed sha discipline, reused.
  * `artifacts/grm_d1/lt1_1/amendment9.json` (GRM contributors, 2026) -- the
    shape of an amendment that binds a completed run's receipts, reused.
  * The predicted-vs-measured table is the house "thresholds registered
    before the gate" rule applied after the fact; no external prior art.
  * Mine: the per-row cause attribution and the flag-set finding.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R2 = Path('artifacts/grm_d1/lt1_1')
R3 = Path('artifacts/grm_f1/lt1_1_r3')
OUT = ROOT / R3

ARMS = {
    'A0': dict(run='run_A0', summary='lead_A0_summary.json', role='CONTROL'),
    "A'": dict(run='run_Aprime', summary='lead_Aprime_summary.json',
               role='TREATMENT'),
    "A+'": dict(run='run_Aplusprime', summary='lead_Aplusprime_summary.json',
                role='TREATMENT'),
}


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha_file(rel):
    return sha_bytes((ROOT / rel).read_bytes())


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def amd_doc(name):
    return json.loads((ROOT / R2 / name).read_text())


def tree_state(section):
    """Amendment 12's blocks, carried forward and RE-VERIFIED unchanged."""
    prior = amd_doc('amendment12.json')[section]
    block = json.loads(json.dumps(prior))
    if section == 'core_rebind':
        for sub, key in (('changed', 'after_sha256'),
                         ('new_inputs', 'sha256'), ('unchanged', 'sha256')):
            for name, entry in sorted(block.get(sub, {}).items()):
                now = sha_file(name)
                if now != entry[key]:
                    raise ValueError(
                        'F1_UNEXPECTED_CORE_DRIFT: %s (amendment 12 pinned '
                        '%s, on tree %s) -- follow-up 4 is ANALYSIS ONLY and '
                        'edits no core file. STOP and report.'
                        % (name, entry[key], now))
        block['supersedes'] = 'amendment12.core_rebind'
        block['attribution_note'] = (
            'CARRIED FORWARD VERBATIM from amendment 12 and re-measured '
            'against the tree. GRM-F1 follow-up 4 is analysis only: it reads '
            'the r3 receipts and writes documents. No core file moved.')
        return block
    for path_key, sha_key in (('path', 'sha256'),
                              ('worker', 'worker_sha256')):
        name = block[path_key]
        now = sha_file(name)
        if now != block[sha_key]:
            raise ValueError(
                'F1_RUNNER_DRIFT: %s (amendment 12 pinned %s, on tree %s) -- '
                'follow-up 4 edits no runner. STOP and report.'
                % (name, block[sha_key], now))
    block['carried_from'] = 'amendment12.json'
    block['change'] = (
        'UNCHANGED since amendment 12. Both shas re-measured against the '
        'tree. Follow-up 4 wrote only REPORT/LEDGER/this amendment and a '
        'test file; the runner and the worker are byte-identical to the '
        'binaries that produced the r3 receipts bound below.')
    return block


def arm_receipts():
    """Bind each arm's summary by sha and read its MEASURED classes."""
    out = {}
    for label, spec in ARMS.items():
        summary_rel = R3 / spec['summary']
        summary = json.loads((ROOT / summary_rel).read_text())
        run_rel = R3 / spec['run']
        cells = sorted(p.name for p in (ROOT / run_rel / 'cells').iterdir()
                       if p.is_dir())
        if len(cells) != 26:
            raise ValueError('F1_R3_CELL_COUNT: %s has %d cells'
                             % (label, len(cells)))
        if not summary.get('complete'):
            raise ValueError('F1_R3_INCOMPLETE: %s' % label)
        measured = {k: '%d/%d' % (v['correct'], v['expected_n'])
                    for k, v in summary['by_class'].items()}
        out[label] = dict(
            role=spec['role'],
            run_dir=str(run_rel),
            summary_path=str(summary_rel),
            summary_sha256=sha_file(summary_rel),
            cells_complete=len(cells),
            complete=bool(summary['complete']),
            alias_fold_merge=summary['binding'].get('alias_fold_merge'),
            measured=measured,
            measured_recalls=summary.get('measured_recalls'))
    return out


#: The eleven rows whose outcome differs across arms, with the cause read off
#: `route_info` in each arm's probes.jsonl. `w` = seat sum of the A' plan.
DIFFERING_ROWS = [
    dict(row='recall_1_25', A0=False, Ap=True, App=True,
         cause='F1 GAIN: retained raw source seats alongside the digest'),
    dict(row='recall_1_50', A0=False, Ap=True, App=True,
         cause='F1 GAIN: same'),
    dict(row='recall_5_25', A0=False, Ap=True, App=True,
         cause="F1 GAIN: correction node 16 plans alone and seats"),
    dict(row='recap_3', A0=False, Ap=True, App=True,
         cause="era index serves; A0's plan head was the wrong node"),
    dict(row='recall_3_10', A0=True, Ap=False, App=True,
         cause="A' plan [12,5,10] = 146 seats vs width 96 -> DROP 10 "
               '(mount_dropped_for_width)'),
    dict(row='recall_5_50', A0=True, Ap=False, App=True,
         cause="A' plan [24,13,32] = 180 -> DROP 32; seated 13+24 mention no "
               'Medibay; the served "12" is the z-coordinate of Breakwater '
               "(-31,48,12) inside node 24 -- a cross-entity numeric leak"),
    dict(row='recall_7_10', A0=True, Ap=False, App=True,
         cause="A' plan [12,5,18] = 35+56+63 = 154 -> DROP 18, the ONLY node "
               'holding the alias relation; admission had IDENTIFIED it. The '
               'abstention is honest: nothing mounted said Beacon IS Lantern'),
    dict(row='recall_7_25', A0=True, Ap=False, App=True, cause='same as 7_10'),
    dict(row='recall_7_50', A0=True, Ap=False, App=True, cause='same as 7_10'),
    dict(row='recall_3_100', A0=False, Ap=False, App=True,
         cause="A+' plans [14] alone (53 seats); A' plan [55,24,14] = 188 "
               'drops both fact-bearing members'),
    dict(row='recap_2', A0=True, Ap=True, App=False,
         cause="A+' plans raw turn 13, whose assistant half is vague; A0/A' "
               'mount the era index whose prose names Iona Vale'),
]

PREDICTION_VERDICTS = {
    'A0': dict(predicted='within +/-1 of r2 arm A on every class',
               measured='fresh 10/15 corr 9/10 alias 10/10 recap 4/5 '
                        '= EXACT match to r2 arm A',
               verdict='MET -- attribution licensed for the treatment arms'),
    "A'": dict(
        fresh=dict(predicted='>= 14/15', measured='11/15 (c2 12)',
                   verdict='REFUTED as stated'),
        corrections=dict(predicted='>= 9/10', measured='9/10', verdict='MET'),
        aliases=dict(predicted='10/10 (must not move at all)',
                     measured='7/10',
                     verdict='FAILED -- the registered falsifier fired'),
        recap=dict(predicted='>= 4/5', measured='5/5', verdict='MET')),
    "A+'": dict(
        fresh=dict(predicted='>= 14/15', measured='13/15 primary, 14/15 c2',
                   verdict='MET on c2, missed on the primary scorer'),
        corrections=dict(predicted='>= 9/10', measured='10/10', verdict='MET'),
        aliases=dict(predicted='10/10', measured='10/10', verdict='MET'),
        recap=dict(predicted='>= 4/5', measured='4/5', verdict='MET')),
}


def finding():
    return dict(
        id='F1-R3-1',
        title='The four flags are a SET at this arena width',
        statement=(
            "A' (F1+F2+F5, A1 OFF) scored aliases 7/10 against the control's "
            '10/10. F1+F2 without A1 is therefore NOT a partial improvement: '
            'it is an alias REGRESSION produced by two individually-correct '
            'treatments interacting.'),
        mechanism=(
            '(1) F2 removes fact-less alias/rename turns from every fold '
            'window -- correct, and it kills the capture defect that IS '
            'present in A0 (digest 24 over sources [13,14,17,18] filed '
            "Commtower's crew and Breakwater's coordinates both onto \"the "
            'Beacon\", and it propagated into the ACTIVE era node 61). '
            '(2) Never folded therefore never superseded: the alias edge '
            "ends A' active=True, digest_of=None, source of nothing -- a "
            'lone raw turn holding a relation and no value. (3) F1 inflates '
            'the rank plan from one member to three. (4) The lone edge is '
            'admitted, ranked third, and evicted for width: 154 seats '
            'planned against a width of 96.'),
        a1_removes_it=(
            'Node 20 carries alias_merge=True, supersedes=[19], and states '
            'relation AND value in ONE 48-seat node, so a single mount '
            "answers the Beacon question. A+' scores aliases 10/10."),
        what_a_prime_would_need=[
            'co-mount of the alias edge WITH its base (63+35 = 98 > 96; '
            'FIX-7 was already STOPPED as impossible at this width)',
            'plan-head protection so an identifier-bearing plan member is '
            'never the one evicted for width',
            'a two-hop read resolving the alias before routing, which the A1 '
            'design deliberately rejected in favour of the write-time join',
        ],
        merge_guidance=(
            "DO NOT merge F1+F2 without A1 on the strength of A'. The "
            'receipts license "the SET A1+F1+F2+F5 beats the control on this '
            'fixture", never a single-flag claim -- four flags moved '
            "together in A+'."),
        evidence=[
            'route_info.mount_dropped_for_width on all five A-prime losses',
            'fold source lists: A0 [13,14,17,18] vs A-prime [13,14]',
            'node 20 metadata alias_merge/supersedes in run_Aplusprime',
        ],
    )


def residency():
    return {
        'A0': dict(nodes=199, active=36, retained_source=0, rows=323,
                   max_summed_token_seats=94),
        "A'": dict(nodes=228, active=213, retained_source=156, rows=322,
                   max_summed_token_seats=95),
        "A+'": dict(nodes=200, active=175, retained_source=113, rows=309,
                    max_summed_token_seats=94),
        'width': 96,
        'bound_exceeded_rows': 0,
        'reading': (
            'The registered residency prediction HOLDS in all three arms. '
            "But A'-s routable base of 213 active nodes against 96 seats "
            'does not overflow the BOUND -- it overflows the PLAN, and the '
            'eviction lands on the node the question named. Residency bounds '
            'were the wrong instrument for this failure; '
            'mount_dropped_for_width was the right one, and it lives only in '
            'route_info, not in the residency rows.'),
    }


def successors():
    return [
        dict(id='F1-R3-S1',
             what='persist fold_guard_history into worker.json',
             why=("F2's window exclusions and attribution rejections cannot "
                  'be counted from the r3 receipts; the finding above had to '
                  'be established from fold SOURCE LISTS instead. A future '
                  'campaign should be able to count them directly.')),
        dict(id='F1-R3-S2',
             what='surface plan-eviction counts in the residency row',
             why=('every A-prime loss is a mount_dropped_for_width event, '
                  'and the residency rows -- the thing the registration '
                  'bounds -- cannot see it.')),
        dict(id='F1-R3-S4',
             what=('persist the column-2 verdict into probes.jsonl and the '
                   'summary'),
             why=('amendment 9 bound c2 as a registered scoring column, but '
                  'the r3 campaign left no receipt for it, so a seat reading '
                  'these artifacts reaches a different total than the '
                  'dispatch reported and cannot audit the column at all.')),
        dict(id='F1-R3-S3',
             what='plan-head protection for identifier-bearing members',
             why=('the narrowest fix that would let an alias-merge-OFF arm '
                  'serve aliases; see finding F1-R3-1.')),
    ]


def amendment13():
    return dict(
        amendment=13,
        arms=sorted(ARMS),
        previous_amendment='amendment12.json',
        previous_amendment_sha256=sha_file(R2 / 'amendment12.json'),
        order='orders/GRM_F1_FOLD_RETAIN_SOURCES.md',
        order_sha256=sha_file('orders/GRM_F1_FOLD_RETAIN_SOURCES.md'),
        purpose=(
            'Bind the completed LT1.1 r3 receipts (three arms, 26/26 cells '
            'each) by sha, record MEASURED results against every registered '
            'prediction, and register the finding that F1+F2 without A1 is '
            'an alias regression at this arena width.'),
        follow_up_instruction=(
            'Amendment 13 supersedes amendment 12 as the governing document. '
            'It changes NO core file and NO runner: both blocks are carried '
            'forward verbatim and re-verified. It is a RESULT amendment.'),
        gpu_executed=True,
        executed_by='the lead, on the card',
        analysed_by='GRM-F1 seat, CPU only, from the committed receipts',
        arm_receipts=arm_receipts(),
        scorer_columns=dict(
            primary=dict(
                scorer='scripts/grm_lt1.score',
                verifiable_from_receipts=True,
                totals={'A0': '33/40', "A'": '32/40', "A+'": '37/40'},
                note=("recomputed by the analysing seat from "
                      "run_*/cells/*/probes.jsonl. A' is a NET REGRESSION "
                      'against the control on this column.')),
            column_2=dict(
                bound_by='amendment9.json',
                verifiable_from_receipts=False,
                reported_by_lead={'A0 fresh': 13, "A' fresh": 12,
                                  "A+' fresh": '14/15',
                                  "A+' total": '38/40'},
                note=('NOT PERSISTED in any r3 artifact: probes.jsonl score '
                      'objects carry exact_correct/category/abstention '
                      'fields and no col2_* key, and lead_*_summary.json '
                      'carries only correct/n/expected_n/exact_rate. These '
                      'figures are RECORDED AS REPORTED and no finding in '
                      'this amendment rests on them.'))),
        registration_bound=dict(
            path=str(R3 / 'registration_merged.json'),
            sha256=sha_file(R3 / 'registration_merged.json')),
        differing_rows=DIFFERING_ROWS,
        differing_row_count=len(DIFFERING_ROWS),
        differing_row_note=(
            'ELEVEN rows differ, not the nine named in the dispatch: '
            'recall_1_25 and recall_1_50 are two further A-prime/A+prime '
            'gains that the class totals absorb.'),
        prediction_verdicts=PREDICTION_VERDICTS,
        beacon_capture=dict(
            A0=dict(nodes_mentioning_beacon=3,
                    capture_present=True,
                    detail=('digest 24 over sources [13,14,17,18]: "the '
                            'maintenance crew of the Beacon is located at '
                            'Iona Vale, and the map position of the Beacon '
                            'is (-31, 48, 12)" -- two other entities\' facts '
                            're-filed onto the alias, propagated into the '
                            'ACTIVE era node 61')),
            **{"A'": dict(nodes_mentioning_beacon=1, capture_present=False,
                          detail=('only the raw alias edge (node 18, '
                                  'active). The same-index digest has '
                                  'sources [13,14] -- alias turns EXCLUDED '
                                  '-- and correctly names Commtower and '
                                  'Breakwater')),
               "A+'": dict(nodes_mentioning_beacon=2, capture_present=False,
                           detail=('retired edge 19 plus A1 merged digest '
                                   '20 (alias_merge=True, supersedes=[19])'))},
            verdict=('F2 ON prevented the capture in BOTH F2 arms; it is '
                     'present only in the control. Verified by a same-window '
                     'fold SOURCE LIST comparison, not by inference.')),
        residency=residency(),
        finding=finding(),
        successors=successors(),
        gate_defect_found=dict(
            id='F1-R3-RED1',
            test=('tests/test_grm_f1_resume_out_root.py::'
                  'test_real_resume_route_through_pending_and_one_run_cell'),
            symptom=('rc == 2 (WORKER_EXIT_1) on one run in ten; all three '
                     'arms on the reproducing run, in 32.78 s vs 102 s'),
            root_cause=(
                'the test patched worker.spawn_argv BEFORE calling resume(); '
                'resume() then enters lt1_1_seams, which assigns '
                'worker.spawn_argv = spawn_argv (the real --worker argv) and '
                'restores its saved value on exit. The seam silently '
                'overwrote the patch, so the leased child took the --worker '
                'branch and loaded the REAL GptOss20B_TC model instead of '
                'the CPU double. Captured traceback: grm_lt1_1.py:880 -> '
                'gpu_loader -> grm_e2e_session.py:2197 '
                'GptOss20B_TC.from_pretrained.'),
            why_it_looked_intermittent=(
                'not a race and not load sensitivity: EVERY run spawned a '
                'real model load, and whether it failed fast enough to '
                'surface as WORKER_EXIT_1 varied with page-cache state. The '
                'tell was the timing -- 102 s for a CPU-double spawn.'),
            fix=('patch runner.spawn_argv, the function the seams INSTALL, '
                 'so the patch survives them. The gate now runs in 15.9 s '
                 'and genuinely exercises the CPU double.'),
            status='ROOT-CAUSED AND FIXED',
            method_note=(
                'five hypotheses were ruled out by check before anyone asked '
                'what the child actually did; the worker.log was on disk the '
                'whole time. "Ruled out by check" is not "diagnosed".'),
            successor=('F1-R3-S5: audit every other test that patches a '
                       'worker.* seam outside lt1_1_seams -- a patch the '
                       'seams overwrite is a test that silently exercises '
                       'production behaviour.')),
        cause_table=dict(
            path='artifacts/grm_f1/r3_cause_table.json',
            sha256=sha_file('artifacts/grm_f1/r3_cause_table.json'),
            rows=11,
            statement=('the per-row receipt for every differing row: mount '
                       'plan, plan seat sum against width, fit_seated, '
                       'mount_dropped_for_width, the admission branch and '
                       'the TEXT of every seated node, per arm. Recomputed '
                       'from probes.jsonl, not transcribed.')),
        report='artifacts/grm_f1/REPORT.md',
        ledger='docs/GRM_F1_FOLD_RETAIN_LEDGER.md',
        runner=tree_state('runner'),
        core_rebind=tree_state('core_rebind'),
        process_safety=dict(
            gpu=('the CAMPAIGN was run by the lead on the card; this seat '
                 'analysed the committed receipts on CPU and ran no GPU work'),
            display='no windowed gate; CPU gates only',
            git='the lead commits; the seat never runs git',
            core_writes='none -- follow-up 4 is analysis only',
            scratch=('pytest uses --basetemp artifacts/grm_f1/tmp and the '
                     'directory is removed afterwards'),
            subagents='none spawned'),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    amd = amendment13()
    payload = canonical(amd)
    digest = sha_bytes(payload)

    chain = ROOT / R2 / 'amendment13.json'
    if chain.exists():
        existing = json.loads(chain.read_text())
        if existing.get('order') != amd['order']:
            raise ValueError(
                'F1_AMENDMENT13_FOREIGN: %s was written by order %r, not by '
                'this one -- refusing to overwrite another order\'s chain '
                'receipt. STOP and report.' % (chain, existing.get('order')))
    chain.write_bytes(payload)
    (ROOT / R2 / 'amendment13.sha256').write_text(
        '%s  amendment13.json\n' % digest)
    (OUT / 'amendment13.json').write_bytes(payload)
    (OUT / 'amendment13.sha256').write_text(
        '%s  amendment13.json\n' % digest)

    for path in (chain, OUT / 'amendment13.json'):
        text = path.read_text()
        for bad in ('/mnt/', '/home/', 'wt/'):
            if bad in text:
                raise ValueError('F1_ABSOLUTE_PIN: %s contains %r'
                                 % (path.name, bad))

    print('amendment13 sha256 %s' % digest)
    for label, rec in sorted(amd['arm_receipts'].items()):
        print('  %-5s %-9s cells=%d alias_merge=%-5s %s'
              % (label, rec['role'], rec['cells_complete'],
                 rec['alias_fold_merge'],
                 ' '.join('%s %s' % (k, v)
                          for k, v in sorted(rec['measured'].items()))))
    print('differing rows bound: %d' % amd['differing_row_count'])
    print('finding: %s' % amd['finding']['title'])
    rb = amd['core_rebind']
    print('core_rebind CARRIED (none moved): %d changed, %d new, %d unchanged'
          % (len(rb['changed']), len(rb['new_inputs']), len(rb['unchanged'])))


if __name__ == '__main__':
    main()
