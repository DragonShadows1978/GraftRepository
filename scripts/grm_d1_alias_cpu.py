#!/usr/bin/env python3
"""GRM-D1 follow-up: the A+ alias probes on the CPU fake session.

This is the D1-side check of A1's claim. It replays the frozen LT1
conversation on the C7 CPU double with `GRM_ALIAS_FOLD_MERGE=1` and
`GRM_ADMISSION_RULE=margin_first` pinned, then asks the five LT1 alias probes
and reports, per probe:

  * did a fold-merge job fire for that entity's alias edge?
  * the digest's FIX-8 identifier set (does it carry BOTH names?)
  * was the digest ADMITTED by routing (is it in the ranking / rank plan)?
  * the mounted node ids
  * the value the CPU double served

No GPU. The reader is `scripts/grm_c7_diagnose.Model` (a regex prose double),
so this measures LINEAGE + ROUTING + ADMISSION, not language-model quality.
The A+ arm's real recall numbers need the registered GPU run; this run's job
is to say whether the alias digest exists, carries both names, and is
reachable by the probe's identifier -- the three things A1 must deliver for
the alias rows to have any chance at all.

Prior art:
  * `core/grm_alias_fold.py` and the `GraftRepository` alias hooks are A1's
    (GRM contributors, 2026), used unchanged through their public surface
    (`alias_fold_pass`, `alias_fold_history`, `alias_fold_pending`). No A1
    internal is reimplemented or monkeypatched here.
  * The flag pin follows R1's `pin_rule` idiom (`scripts/grm_r1_replay.py:242`,
    GRM contributors 2026): pin AFTER the ambient `GRM_*` strip, then READ
    BACK and assert, so an arm can never silently run the default. Taken: the
    pin-then-verify contract. Mine: applying it to `GRM_ALIAS_FOLD_MERGE`
    alongside the admission rule, and asserting the readback through A1's own
    `alias_fold_enabled`.
  * The CPU replay harness (`repository`, `Codec`, `Model`, the visible-prose
    reader) is C7's (`scripts/grm_c7_diagnose.py`) and LT1's
    (`scripts/grm_lt1_cpu.py`), reused unchanged.
  * The identifier projection for the "digest names both" check is A1's own
    `grm_alias_fold.identifier_set`, deliberately the SAME projection routing
    and admission see, so the check is not a private notion of "contains".
  * Mine: the per-probe join of (fold decision, digest identifiers, admission
    ranking, mount, served value) into one row, and the A/A+ contrast.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_d1'
FIXTURE = ROOT / 'fixtures/lt1/dialogue.json'

ALIAS_ENV = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'


def _lt1_arm_a_alias_outcomes():
    """How each alias probe actually scored on the LT1 GPU run, arm A.

    Read from D1's cause table (built from the frozen receipts) rather than
    typed, so the contrast column cannot drift from the measurement. Probes
    absent from the table were CORRECT -- the table lists wrong rows only.
    """
    table = OUT / 'cause_table_A.json'
    wrong = set()
    if table.exists():
        wrong = {r['question_id'] for r in json.loads(table.read_text())['rows']}
    fixture = json.loads(FIXTURE.read_text())
    return {p['id']: ('**WRONG**' if p['id'] in wrong else 'correct')
            for p in fixture['probes'] if p['class'] == 'alias'}


LT1_ARM_A_ALIAS = _lt1_arm_a_alias_outcomes()


def pin_arm(patch, *, alias_fold):
    """Pin both arm variables and READ BACK, R1's `pin_rule` contract.

    Prior art: scripts/grm_r1_replay.py:242 `pin_rule` (GRM contributors,
    2026). The registered admission rule is the same for both arms; the ONLY
    difference between A and A+ is `GRM_ALIAS_FOLD_MERGE`.
    """
    from core.grm_admission import admission_rule
    from core.grm_alias_fold import alias_fold_enabled

    patch.setenv(RULE_ENV, 'margin_first')
    if alias_fold:
        patch.setenv(ALIAS_ENV, '1')
    else:
        patch.delenv(ALIAS_ENV, raising=False)

    observed_rule = admission_rule()
    if observed_rule != 'margin_first':
        raise ValueError('D1_RULE_PIN_FAILED: observed=' + observed_rule)
    observed_alias = alias_fold_enabled()
    if observed_alias is not bool(alias_fold):
        raise ValueError('D1_ALIAS_PIN_FAILED: wanted=%s observed=%s'
                         % (bool(alias_fold), observed_alias))
    return dict(admission_rule=observed_rule, alias_fold_merge=observed_alias)


def _open(path, *, alias_fold):
    """A CPU repository with the A/A+ switch pinned and verified.

    `GraftRepository` freezes the alias choice at construction time on
    purpose, so the environment is pinned BEFORE the constructor runs and the
    constructor's own resolution is then asserted against the pin.
    """
    import pytest
    from scripts.grm_c7_diagnose import repository
    patch = pytest.MonkeyPatch()
    pinned = pin_arm(patch, alias_fold=alias_fold)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    repo = repository(path / 'repository', patch)
    # The C7 helper sets its own env; re-pin and re-verify after it, then
    # assert the repository actually resolved the flag we asked for.
    pinned = pin_arm(patch, alias_fold=alias_fold)
    if bool(repo.alias_fold_merge) is not bool(alias_fold):
        raise ValueError('D1_REPO_ALIAS_RESOLUTION_MISMATCH: wanted=%s got=%s'
                         % (bool(alias_fold), repo.alias_fold_merge))
    repo.arena.width = 96            # arm A geometry (profile)
    repo.arena.recency_mounts = 2
    return repo, patch, pinned


def run(path, *, alias_fold, upto=None):
    """Replay the frozen conversation, then ask the five alias probes."""
    from core import grm_alias_fold as af
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c7_diagnose import Model
    from scripts.grm_lt1_cpu import visible_answer

    fixture = json.loads(FIXTURE.read_text())
    alias_probes = [p for p in fixture['probes'] if p['class'] == 'alias']

    repo, patch, pinned = _open(path, alias_fold=alias_fold)
    arena = repo.arena
    turn_nodes = {}
    replayed = 0

    # The CPU double must answer TWO different kinds of request, and the
    # original LT1 ProseModel only answers one of them.
    #
    #   * a PROBE read -> the visible-prose answer (LT1's arm, unchanged);
    #   * a CONSOLIDATION generation -> a digest that preserves the source
    #     facts. `ArenaCache.consolidate` gates the digest on
    #     `_coverage(text, _fact_set(sources))`, so a stub that says
    #     "unknown" scores 0.0 and EVERY fold correctly aborts with
    #     `alias_fold_fidelity_abort`. That abort is the C7 double's limit,
    #     not an A1 defect -- see the RED note in the report.
    #
    # For the fold path we therefore supply a faithful EXTRACTIVE digest: the
    # source user-sentences concatenated. It invents nothing, and it is the
    # same shape `_extractive_era_text` already produces for deep folds, so
    # the coverage gate is exercised rather than bypassed.
    #
    # Prior art: scripts/grm_lt1_cpu.py ProseModel (GRM contributors, 2026)
    # for the probe half; core/graft_arena.py `_extractive_era_text` for the
    # digest shape. Mine: routing the two request kinds to the two readers.
    from core.grm_alias_fold import alias_scan_text

    consolidating = {'on': False, 'sources': []}

    class DualModel(Model):
        def __call__(self, ids, kv_caches=None, position_offset=0, **kwargs):
            if kv_caches is None:
                text = self.codec.decode(ids[0])
                if consolidating['on']:
                    self.fold_output = (
                        ' '.join(consolidating['sources']) + '<|end|>')
                else:
                    seen = ((self.injected if arena.cur_mounts else '')
                            + '\n' + text)
                    self.fold_output = visible_answer(seen, text)
            return super().__call__(ids, kv_caches, position_offset, **kwargs)
    arena.m.__class__ = DualModel

    # Flag the consolidation window without touching A1 or the arena: wrap
    # `consolidate` so the double knows which reader to use and what the
    # sources were. No A1 decision, no coverage gate, is altered.
    original_consolidate = arena.consolidate

    def consolidate(idxs, *args, **kwargs):
        consolidating['on'] = True
        consolidating['sources'] = [
            ' '.join(alias_scan_text(arena.grafts[i].get('text', '')).split())
            for i in idxs]
        try:
            return original_consolidate(idxs, *args, **kwargs)
        finally:
            consolidating['on'] = False
    arena.consolidate = consolidate

    try:
        stop = upto or 200
        for event in fixture['turns']:
            if event['turn'] > stop:
                break
            if event['kind'] in ('probe', 'recap'):
                continue
            idx = arena.feed(e2e.harmony_turn(event['user'], event['assistant']))
            arena.grafts[idx]['kind'] = 'turn'
            turn_nodes[event['turn']] = idx
            replayed += 1

        # Deposit-time folding is a runtime-funnel hook; this replay feeds the
        # arena directly (as LT1's own CPU arm does), so drive A1's PUBLIC
        # librarian sweep for already-stored aliases instead -- the same entry
        # point a resumed repository uses.
        pending_before = repo.alias_fold_pending()
        merges = repo.alias_fold_pass()
        pending_after = repo.alias_fold_pending()

        rows = []
        for probe in alias_probes:
            entity = probe['entity']
            edge = next(t for t in fixture['turns']
                        if t.get('kind') == 'alias' and t.get('entity') == entity)
            alias_name = edge['alias']
            edge_node = turn_nodes.get(edge['turn'])
            decisions = [d for d in repo.alias_fold_history
                         if d.get('edge') == edge_node]
            merged = [d for d in decisions if d.get('reason') == af.REASON_MERGED]

            # The digest, if one exists: an ACTIVE alias digest naming BOTH the
            # alias and the base, by A1's own identifier projection.
            digests = [i for i, g in enumerate(arena.grafts)
                       if not g.get('retired')
                       and repo._is_alias_digest(i)
                       and af.names_present(g.get('text', ''), alias_name, entity)]

            answer, info = e2e._probe_ladder_chat(
                repo, probe['question'], topk=3, ngen=32, max_trips=1,
                defer_memory=True)
            ranking = [int(x) for x in (info.get('ranking_ids') or [])]
            seated = [int(x) for x in (info.get('fit_seated') or [])]
            admitted = bool(set(digests) & set(ranking)) if digests else False

            rows.append(dict(
                probe_id=probe['id'], distance=probe['distance'],
                entity=entity, alias=alias_name,
                question=probe['question'], expected=probe['expected'],
                edge_turn=edge['turn'], edge_node=edge_node,
                fold_job_fired=bool(merged),
                fold_decisions=decisions,
                digest_nodes=digests,
                digest_identifiers=(sorted(af.identifier_set(
                    arena.grafts[digests[0]].get('text', ''))) if digests else []),
                digest_names_both=bool(digests),
                admitted=admitted,
                digest_mounted=bool(set(digests) & set(seated)),
                admission_rank_plan=info.get('admission_rank_plan'),
                admission_identified_candidates=info.get(
                    'admission_identified_candidates'),
                ranking_ids=ranking,
                mounted_ids=seated,
                served=str(answer),
                served_correct=probe['expected'] in str(answer),
            ))
        return dict(
            arm='A+' if alias_fold else 'A',
            pinned=pinned,
            turns_replayed=replayed,
            alias_fold_pending_before=pending_before,
            alias_fold_pending_after=pending_after,
            merges_executed=len(merges),
            alias_fold_history=list(repo.alias_fold_history),
            rows=rows,
            evidence_class='CPU fake session (lineage/routing/admission only; '
                           'NOT a language-model quality measurement)')
    finally:
        patch.undo()
        repo.close()


def markdown(plus, base):
    out = ['# GRM-D1 — A+ alias probes on the CPU fake session', '',
           '`%s=1`, `%s=margin_first` (both pinned and read back, R1 idiom), '
           'arena width 96, %d turns replayed.'
           % (ALIAS_ENV, RULE_ENV, plus['turns_replayed']), '',
           'Evidence class: %s' % plus['evidence_class'], '',
           'Alias fold jobs pending before the sweep: %d; merges executed: %d; '
           'pending after: %d.' % (plus['alias_fold_pending_before'],
                                   plus['merges_executed'],
                                   plus['alias_fold_pending_after']), '',
           '| probe | d | alias -> base | LT1 GPU arm A | fold job fired? | digest node | digest identifiers (FIX-8) | admitted? | mounted ids | served value (CPU double) |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in plus['rows']:
        ident = ', '.join('`%s`' % t for t in r['digest_identifiers'][:10])
        if len(r['digest_identifiers']) > 10:
            ident += ', …(%d total)' % len(r['digest_identifiers'])
        out.append('| `%s` | %d | %s -> %s | %s | %s | %s | %s | %s | %s | `%s` |' % (
            r['probe_id'], r['distance'], r['alias'], r['entity'],
            LT1_ARM_A_ALIAS.get(r['probe_id'], '?'),
            '**yes**' if r['fold_job_fired'] else 'no',
            r['digest_nodes'] or '—', ident or '—',
            '**yes**' if r['admitted'] else 'no',
            r['mounted_ids'], r['served'].replace('|', '\\|')[:60]))
    out += ['', '## Arm A (flag OFF) contrast', '',
            '| probe | fold job fired? | digest node | admitted? | mounted ids | served value |',
            '|---|---|---|---|---|---|']
    for r in base['rows']:
        out.append('| `%s` | %s | %s | %s | %s | `%s` |' % (
            r['probe_id'], 'yes' if r['fold_job_fired'] else 'no',
            r['digest_nodes'] or '—', 'yes' if r['admitted'] else 'no',
            r['mounted_ids'], r['served'].replace('|', '\\|')[:60]))
    out += ['', '## Counts', '',
            '- A+ fold jobs fired: %d/%d probes'
            % (sum(1 for r in plus['rows'] if r['fold_job_fired']), len(plus['rows'])),
            '- A+ digests naming BOTH names: %d/%d'
            % (sum(1 for r in plus['rows'] if r['digest_names_both']), len(plus['rows'])),
            '- A+ digests ADMITTED by routing: %d/%d'
            % (sum(1 for r in plus['rows'] if r['admitted']), len(plus['rows'])),
            '- A+ digests MOUNTED: %d/%d'
            % (sum(1 for r in plus['rows'] if r['digest_mounted']), len(plus['rows'])),
            '- A  fold jobs fired: %d/%d (flag OFF: must be 0)'
            % (sum(1 for r in base['rows'] if r['fold_job_fired']), len(base['rows'])),
            '']
    return '\n'.join(out) + '\n'


def main():
    import tempfile
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        plus = run(Path(tmp) / 'aplus', alias_fold=True)
        base = run(Path(tmp) / 'a', alias_fold=False)
    payload = {'A+': plus, 'A': base}
    (OUT / 'alias_cpu_aplus.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    (OUT / 'alias_cpu_aplus.md').write_text(markdown(plus, base))
    for row in plus['rows']:
        print('%-14s fold=%-5s digest=%-8s admitted=%-5s mounted=%-10s served=%r'
              % (row['probe_id'], row['fold_job_fired'], row['digest_nodes'],
                 row['admitted'], row['mounted_ids'], row['served'][:44]))
    n = len(plus['rows'])
    print('A+ fired %d/%d  admitted %d/%d  mounted %d/%d  served-correct %d/%d'
          '   |  A fired %d/%d served-correct %d/%d'
          % (sum(1 for r in plus['rows'] if r['fold_job_fired']), n,
             sum(1 for r in plus['rows'] if r['admitted']), n,
             sum(1 for r in plus['rows'] if r['digest_mounted']), n,
             sum(1 for r in plus['rows'] if r['served_correct']), n,
             sum(1 for r in base['rows'] if r['fold_job_fired']), n,
             sum(1 for r in base['rows'] if r['served_correct']), n))


if __name__ == '__main__':
    main()
