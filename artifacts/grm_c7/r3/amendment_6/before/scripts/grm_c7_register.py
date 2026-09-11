#!/usr/bin/env python3
"""Write C7 registration ONCE, before gates.

Prior art: EB1/C2 registration and session plants (GRM contributors, 2026).
Taken: exact-source fixtures, immutable inputs, timing receipts and arm flags.
C7 contribution: deterministic collision-free placement and costed ranges;
no prior art known to me for this fixture. Greedy scheduling is established
practice, not a novelty claim.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.grm_c7_common import ROOT, OUT, FIX, REG, DISTANCES, CLASSES, create, read, sha


def fixture():
    events = {}
    probes = []
    entities = set()
    aliases = []
    corrections = []
    forced_folds = []

    def fact(t, entity, value, text=None, **kw):
        entities.add(entity)
        phrase = f'current {entity} value is {value}'
        events[t] = {'turn': t, 'kind': 'fact', 'fact_id': entity, 'value': value,
                     'fact_phrase': phrase, 'user': text or f'The {phrase}.',
                     'assistant': 'Recorded.', **kw}

    # Supporting sources precede every anchor. Correction history is verbatim
    # in the oracle, including superseded values; expected answers never enter
    # the memory query. Fresh fixture namespace is unrelated to C2 needles.
    fact(1, 'C7-Vesper-0', 'Mica-431',
         'The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.')
    entities.add('C7-Vesper-1')
    fact(2, 'C7-Vesper-0', 'Flint-511',
         'Correction of both original records: the current C7-Vesper-0 value is Flint-511; '
         'the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.')
    fact(3, 'C7-Archive-0', 'Reed-611', 'For C7-Archive-0 the outbound tag is Reed-611.')
    fact(4, 'C7-Archive-1', 'Reed-612', 'For C7-Archive-1 the outbound tag is Reed-612.')
    fact(5, 'C7-AliasBase-0', 'Jasper-711',
         'The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.')
    entities.add('C7-AliasBase-1')
    occupied = set(events)
    anchors = []
    for cls in CLASSES:
        for j in range(3):
            anchor = next(t for t in range(6, 50)
                          if not ({t, *(t+d for d in DISTANCES)} & occupied))
            occupied.update({anchor, *(anchor+d for d in DISTANCES)})
            anchors.append((cls, j, anchor))
    for cls, j, anchor in anchors:
        answerable = j < 2
        if cls == 'fresh':
            entity = f'C7-Fresh-{j}'
            value = f'Basalt-{811+j}'
            fact(anchor, entity, value)
            source = [anchor]
            question = f'What is the current {entity} value?'
        elif cls == 'alias':
            entity = f'C7-AliasBase-{j}'
            value = f'Jasper-{711+j}'
            alias = f'C7-Signal-{j}'
            fact(anchor, entity, value if answerable else 'UNSET',
                 f'{alias} is an alias for {entity}.')
            aliases.append({'turn': anchor, 'alias': alias, 'entity': entity})
            source = [5, anchor]
            question = f'What is the current value for {alias}?'
        elif cls == 'correction':
            entity = f'C7-Vesper-{j}'
            value = f'Onyx-{911+j}'
            fact(anchor, entity, value)
            events[anchor].update(kind='supersede', old_value=f'Flint-{511+j}',
                correction_command=f'correct memory: current {entity} value is Flint-{511+j} => '
                                   f'The current {entity} value is {value}.')
            corrections.append({'turn': anchor, 'entity': entity, 'corrects_turn': 2,
                                'earlier_correction': True})
            if not answerable:
                # Negative control introduces an entity, not a fictitious
                # correction of an unregistered predecessor.
                events[anchor]['kind'] = 'fact'
                events[anchor].pop('correction_command')
                events[anchor].pop('old_value')
                corrections.pop()
            source = [1, 2, anchor]
            question = f'After all corrections, what is the current {entity} value?'
        else:
            entity = f'C7-Archive-{j}'
            value = f'Reed-{611+j}|Cedar-{1011+j}'
            fact(anchor, entity, f'Cedar-{1011+j}',
                 f'For {entity} the inbound tag is Cedar-{1011+j}.')
            source = [3+j, anchor] if answerable else [anchor]
            question = f'For {entity}, give outbound then inbound tags separated by |.'
            if answerable:
                forced_folds.append({'after_turn': anchor, 'source_turns': source,
                                     'kind': 'digest', 'retry': False})
        if not answerable:
            question = f'What is the inspection password for {entity}?'
        for distance in DISTANCES:
            t = anchor + distance
            pid = f'c7_{cls}_{j}_d{distance:03d}'
            probe = {'id': pid, 'turn': t, 'class': cls, 'distance': distance,
                     'source_turn': anchor, 'source_turns': source,
                     'entity': entity, 'answerable': answerable,
                     'expected': value if answerable else 'UNKNOWN',
                     'question': question + ' Reply only with the answer; if unspecified, reply UNKNOWN.',
                     'requires_folded': cls == 'folded' and answerable}
            probes.append(probe)
            events[t] = {'turn': t, 'kind': 'probe', 'user': probe['question'],
                         'fact_id': entity, 'source_turn': anchor,
                         'expected': probe['expected'], 'accepts': [probe['expected']],
                         'probe_id': pid}

    # New unrelated entities, 12 more aliases, and six complete correction
    # chains. No answer values are repeated by later unrelated filler turns.
    free = iter(t for t in range(50, 251) if t not in events)
    for j in range(40):
        entity = f'C7-Outpost-{j:02d}'
        t = next(free)
        fact(t, entity, f'Slate-{1200+j}')
        if j < 12:
            aliases.append({'turn': t, 'entity': entity, 'alias': f'C7-Handle-{j:02d}'})
            events[t]['user'] += f' C7-Handle-{j:02d} is an alias for {entity}.'
        if j < 6:
            last_t = t
            for generation, prefix in enumerate(('Coral', 'Pearl'), 1):
                t = next(free)
                old = f'{"Slate" if generation == 1 else "Coral"}-{1200+j}'
                fact(t, entity, f'{prefix}-{1200+j}')
                events[t].update(kind='supersede', old_value=old,
                    correction_command=f'correct memory: current {entity} value is {old} => '
                                       f'The current {entity} value is {prefix}-{1200+j}.')
                corrections.append({'turn': t, 'entity': entity, 'corrects_turn': last_t,
                                    'earlier_correction': generation == 2})
                last_t = t
    for t in range(1, 301):
        events.setdefault(t, {'turn': t, 'kind': 'filler',
            'user': f'Log marker {t}. Discuss briefly how to arrange empty shipping crates; '
                    'do not invent or change any named record values.'})
    for p in probes:
        p['oracle_source_texts'] = [events[t]['user'] for t in p['source_turns']]
    return {'schema': 'grm.c7.fixture.v1', 'turn_numbering': 'one-based, inclusive',
            'distance_definition': 'probe turn minus latest authoritative source/alias/negative anchor turn',
            'entities': sorted(entities), 'aliases': aliases, 'corrections': corrections,
            'turns': [events[t] for t in range(1, 301)],
            'probes': sorted(probes, key=lambda p: p['turn']), 'forced_folds': forced_folds,
            'restart_after_turns': [100, 200], 'pressure_after_turns': [60, 140, 240]}


def register():
    if REG.exists():
        raise ValueError('Registration already exists: amendment required')
    f = fixture()
    create(FIX, f)
    create(OUT/'fixture_manifest.json', {'schema': 'grm.c7.fixture-manifest.v1',
        'fixture_sha256': sha(FIX), 'turns': 300, 'probes': len(f['probes']),
        'entities': len(f['entities']), 'aliases': len(f['aliases']),
        'corrections': len(f['corrections'])})
    timing_path = Path('/mnt/ForgeRealm/GraftRepository/artifacts/grm_eb1/g5_run/lh-5/session/instrumentation.jsonl')
    rows = [__import__('json').loads(line) for line in timing_path.read_text().splitlines()]
    walls = [row['turn_wall_ms']/1000 for row in rows]
    mean = sum(walls)/len(walls)
    probe_max = max(row['turn_wall_ms']/1000 for row in rows if row['kind'] == 'probe')
    correction_max = max(row['turn_wall_ms']/1000 for row in rows if row['kind'] == 'supersede')
    # Prior art: C2 measured-wall cell scheduling (GRM, 2026). Ours: budget
    # oracle generation, two-sided restart echoes and explicit fold pressure.
    # 1.25 is an unmeasured safety factor, not a measured performance result.
    base, oracle = 1.25*mean, 1.25*probe_max
    nominal = math.floor((280-60-60)/(base+oracle))
    weights = {}
    for event in f['turns']:
        t = event['turn']
        w = max(base, 1.25*correction_max) if event['kind'] == 'supersede' else base
        if event['kind'] == 'probe':
            w += oracle
        if t in [v['after_turn'] for v in f['forced_folds']]:
            w += 60
        if t in f['pressure_after_turns']:
            w += 20
        if t in (100, 101, 200, 201):
            w += 4*oracle
        weights[t] = w
    cells = []
    for arm in ('A', 'B'):
        start = 1
        previous = None
        while start <= 300:
            stop = start-1
            cost = 60.0  # 40 s model/repository reload + 20 s durable boundary IO, heuristic
            while stop < 300 and stop-start+1 < nominal:
                if (stop >= start and stop in (100, 200)) or cost+weights[stop+1] > 280:
                    break
                stop += 1
                cost += weights[stop]
            if stop < start:
                raise ValueError(f'NON_FIT_SINGLE_TURN: {start}')
            cid = f'{arm}-{start:03d}-{stop:03d}'
            cells.append({'id': cid, 'arm': arm, 'start': start, 'stop': stop,
                          'depends': previous, 'estimate_seconds': round(cost, 6),
                          'worker_seconds': 280, 'lease_seconds': 285,
                          'outer_seconds': 590, 'cooldown_seconds': 30})
            previous, start = cid, stop+1
    regpath = ROOT/'config/grm_eb1_profile_registered.json'
    profile = read(regpath)
    costs = {arm: sum(c['estimate_seconds'] for c in cells if c['arm'] == arm) for arm in ('A','B')}
    create(OUT/'timing.json', {'evidence_class': 'historical EB1 E2E receipt; not C7 benchmark',
        'source': str(timing_path), 'sha256': sha(timing_path), 'deduplication':
        'only final cumulative lh-5 receipt; earlier cumulative checkpoints excluded',
        'n_turns': len(walls), 'mean_turn_seconds': mean, 'max_turn_seconds': max(walls),
        'max_probe_seconds': probe_max, 'max_correction_seconds': correction_max,
        'safety_factor': 1.25, 'reload_io_allowance_seconds_per_cell': 60,
        'forced_fold_allowance_seconds': 60, 'pressure_allowance_seconds': 20,
        'nominal_turns_per_cell': nominal,
        'derivation': 'floor((280-60 reload/IO-60 fold reserve)/(1.25*mean turn +1.25*max probe))',
        'actual_cell_rule': 'at most nominal turns, greedily shorten to <=280s and boundaries 100/200; corrections use historical max',
        'uncertainty': 'reload/IO, extra folds and pressure unmeasured; B width256 uses A proxy, no measured B speed claim'})
    files = [ROOT/'orders/GRM_C7_300_TURN_RESIDENCY.md', regpath, FIX,
             OUT/'fixture_manifest.json', OUT/'timing.json', timing_path,
             Path('/mnt/Shared/HOUSE_RULES.md'),
             Path('/mnt/ForgeRealm/wt/grm-c3/orders/GRM_C3_DNGH_DECOY_CALIBRATION.md')]
    files += [Path('/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so'),
              Path(profile['model']['path'])/'config.json',
              Path(profile['model']['path'])/'tokenizer.json']
    files += [p for d in ('core','scripts','config','tests') for p in (ROOT/d).rglob('*')
              if p.is_file() and p.suffix in ('.py','.json')]
    inputs = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in sorted(set(files))}
    create(REG, {'schema': 'grm.c7.registration.v1', 'immutable': True,
        'order': 'orders/GRM_C7_300_TURN_RESIDENCY.md', 'model': profile['model'],
        'model_frame': profile['frame'], 'agent_model': 'GPT-6 (exact deployment ID not exposed)',
        'agent_effort': 'high requested by order; deployment setting not independently exposed',
        'budget_seconds_arm_A': 7200, 'immutable_inputs': inputs, 'cells': cells,
        'prediction': 'Residency stays bounded, but at least one attribution/admission/fold failure appears beyond repeated needles.',
        'arms': {a: {'flags': dict(profile['default_flags'], **(profile['flag_overrides'] if a == 'A' else {})),
                     'estimate_seconds': costs[a], 'gpu_hours': costs[a]/3600,
                     'status': 'FIT_ESTIMATE' if (costs[a] if a == 'A' else sum(costs.values())) <= 7200 else 'NON_FIT',
                     'standalone_fit_estimate': costs[a] <= 7200,
                     'cost_basis': 'same EB1 proxy; B width256 speed unmeasured'} for a in ('A','B')},
        'B_within_combined_7200_status': 'NON_FIT' if sum(costs.values()) > 7200 else 'FIT_ESTIMATE',
        'B_execution': 'NON_FIT against remaining campaign cap; immutable prospective cells retained, never launched or retried under this registration',
        'probe_protocol': 'Probe events occupy script turns but defer memory deposit, so query answers cannot refresh source distance. Both arms identical; exact-source oracle is isolated and never deposited.',
        'acceptance': {'max_exact_answer_error_delta_per_distance': 0.15,
                       'max_seats_rule': 'each served phase summed mounted ntok <=192 + actual recency-only mounted ntok',
                       'restart': 'same per-probe exact/wrong/abstention scores at 100/200, same persisted metadata, different process identity',
                       'coverage': 0.70, 'complete_rows_required': 60,
                       'folded': 'all 10 answerable folded probes must seat digest/era covering sources with no raw source mount',
                       'paging': 'each registered pressure evicts >=1 old payload and a later natural mount reloads it from NVMe',
                       'empty_missing_or_partial': 'NOT_RUN/INCOMPLETE, never PASS'},
        'cpu_gates': ['fixture integrity and disjoint source/probe placement',
                      '3 probes in each distance/class cell including one unspecified attribute',
                      'exact oracle scorer hand cases incl wrong+right, old correction, abstention',
                      'synthetic state persisted then reloaded in a new process at a real cell boundary',
                      'corruption, missing rows, fold/raw leakage and budget guards reject'],
        'mutations': {'threshold': .80, 'only_after_passing_baseline': True,
                      'ids': ['substring_instead_of_exact','erase_abstention_error',
                              'erase_unsupported_error','invert_exact_failure',
                              'overlap_wrong_and_abstention'],
                      'scope': 'CPU scorer source copies; not blind review'},
        'restart_sentinels': [next(p['id'] for p in f['probes'] if p['class']==c and p['answerable']) for c in CLASSES],
        'pressure': {'after_turns': f['pressure_after_turns'], 'old_age_turns': 30,
                     'method': 'flush; set instance vram_budget=0; production _page; restore budget; drop old host copies only after NPZ durability check; natural node_loader reload observed',
                     'same_intervention_both_arms': True},
        'fold_failures': 'append every attempt/source text/result immediately including exceptions; capture full decoded candidate text before 1200-character product truncation; no retries or threshold changes',
        'restart_count_caveat': 'two scored stress restarts; every leased cell also reloads in a new process',
        'stop_rule': 'no retry of started/RED/NON_FIT cell; reject changed inputs; reserve 280 seconds against remaining 7200 budget before each cell; stop on timeout or incomplete durability',
        'prior_art': 'Local EB1, C2, RS3, S4, SCOUT-FIX-1 (GRM contributors, 2026), verified source: serving, full-text live ceiling, lineage, coverage guard, LRU pager and immutable snapshots. C7 adds fixtures, instrumentation and strict scoring, no new memory algorithm; no prior art known to me for this exact combined protocol.'})
    with (OUT/'registration.sha256').open('x') as stream:
        stream.write(f'{sha(REG)}  registration.json\n')
    print({'registration_sha256': sha(REG), 'nominal_turns_per_cell': nominal,
           'cells': len(cells), 'cost_seconds': costs})


if __name__ == '__main__':
    register()
