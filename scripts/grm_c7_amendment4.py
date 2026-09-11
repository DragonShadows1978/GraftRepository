#!/usr/bin/env python3
"""CPU receipt replay and oracle-only prompt proposal; never loads a model.

Prior art: GRM C7 receipts/scorer, EB1 harmony_turn, SC1 glyph normalization,
FIX-3 coverage/QC and SHA-bound amendments (GRM contributors, 2026), verified
local source. Reuse their exact functions via AST extraction, not new scoring.
Our contribution is this audit and source-only latest-field presentation.
Chronological overwrite is an ordinary latest-write rule; no prior art known
to me for this exact C7 diagnostic rendering. No new memory algorithm.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / 'artifacts/grm_c7/r2/amendment_4/registration.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_functions():
    # Prior art: Python AST (Python contributors, 2026), local GRM pure
    # functions. Extract unchanged function bodies to avoid importing the
    # CUDA/model runtime. AST execution here is CPU analysis, not inference.
    ns = {'re': re, 'unicodedata': unicodedata}
    for name, funcs, constants in [
        ('scripts/grm_c7_common.py', {'normalize', 'score'}, set()),
        ('scripts/grm_e2e_session.py', {'harmony_turn'},
         {'HARMONY_SINK', 'SYSTEM_PREFIX', 'ASSISTANT_FINAL', 'HARMONY_STOPS'}),
        ('core/grm_text_norm.py', {'normalize_glyphs'}, {'UNICODE_HYPHENS', '_EMPHASIS'}),
    ]:
        tree = ast.parse((ROOT / name).read_text())
        nodes = [n for n in tree.body if
                 isinstance(n, ast.FunctionDef) and n.name in funcs or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                     t.id in constants for t in n.targets)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), ns)
    tree = ast.parse((ROOT / 'core/graft_arena.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArenaCache')
    names = {'_rare_tokens', '_caps_tokens', '_fact_set', '_coverage', '_digest_qc'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id in {'_FACT_STOP', 'DIGEST_PROMPTS', 'ERA_PROMPTS'} for t in n.targets)]
    cpu = ast.ClassDef(name='ArenaCache', bases=[], keywords=[], body=nodes,
                       decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cpu], type_ignores=[])),
                 'core/graft_arena.py:pure_extract', 'exec'), ns)
    # Recorded campaign has GRM_LSR_FIXES=1. Use its exact glyph projection.
    ns['ArenaCache']._norm_text = staticmethod(ns['normalize_glyphs'])
    return ns


def latest_lines(records):
    # Prior art: local C7 chronological-source oracle (GRM, 2026); ordinary
    # latest-write overwrite. Ours: C7 grammar-specific, source-only rendering
    # for an oracle diagnostic. No prior art known to me for this exact
    # rendering. This function has no expected/answerable/query argument.
    fields, aliases = {}, {}
    for turn, source in sorted(records):
        for m in re.finditer(r'current (C7-[A-Za-z0-9-]+) value is ([A-Za-z0-9-]+)', source):
            fields[(m[1], 'value')] = (m[2], turn, source[m.start():m.end()])
        for m in re.finditer(r'(C7-[A-Za-z0-9-]+) is an alias for (C7-[A-Za-z0-9-]+)', source):
            aliases[m[1]] = (m[2], turn, source[m.start():m.end()])
        for m in re.finditer(r'For (C7-[A-Za-z0-9-]+) the (outbound|inbound) tag is ([A-Za-z0-9-]+)', source):
            fields[(m[1], m[2] + ' tag')] = (m[3], turn, source[m.start():m.end()])
    result = []
    for (entity, field), (value, turn, quote) in sorted(fields.items()):
        result.append({'line': f'{entity}: latest {field} = {value} [turn {turn}].',
                       'provenance': [{'turn': turn, 'quote': quote}]})
    for alias, (base, turn, quote) in sorted(aliases.items()):
        result.append({'line': f'{alias}: alias of {base} [turn {turn}].',
                       'provenance': [{'turn': turn, 'quote': quote}]})
        if (base, 'value') in fields:
            value, vt, vq = fields[(base, 'value')]
            result.append({'line': f'{alias}: latest value through {base} = {value} '
                           f'[value turn {vt}; alias turn {turn}].',
                           'provenance': [{'turn': vt, 'quote': vq},
                                          {'turn': turn, 'quote': quote}]})
    return result


def trace_attempts(texts, stops, budget=120):
    # Prior art: FIX-3's any(stop in decode(out)) loop (GRM, 2026), exact
    # local source. Reconstruct call counts from its recorded decode strings:
    # four checks per non-stop step, short-circuit at a matching stop, then
    # one final decode. No tokenizer re-encoding estimate is substituted for
    # generated-token count. No prior art known to me for this audit parser.
    cursor, attempts = 0, []
    for _ in range(3):
        found = None
        for n in range(1, budget):
            raw = texts[cursor]
            for stop in stops:
                assert texts[cursor] == raw
                cursor += 1
                if stop in raw:
                    found = stop
                    break
            if found:
                break
        else:
            n = budget
        final = texts[cursor]
        cursor += 1
        if found:
            assert final == raw
        attempts.append({'generated_tokens_from_decode_trace': n,
                         'stop': found or next((s for s in stops if s in final), None),
                         'budget_exhausted': n == budget,
                         'raw_final_decode': final})
    assert cursor == len(texts), (cursor, len(texts))
    return attempts


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '&#124;').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                     '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                    ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def run(output):
    reg = read(REG)
    assert sha(REG) == REG.with_suffix('.sha256').read_text().split()[0]
    amendment_path = REG.parent / 'evidence_amendment_1.json'
    amendment = read(amendment_path)
    assert sha(amendment_path) == amendment_path.with_suffix('.sha256').read_text().split()[0]
    assert amendment['previous_registration_sha256'] == sha(REG)
    for name, digest in amendment['additional_inputs'].items():
        assert sha(ROOT / name) == digest
    amendment2_path = REG.parent / 'evidence_amendment_2.json'
    assert sha(amendment2_path) == amendment2_path.with_suffix('.sha256').read_text().split()[0]
    assert read(amendment2_path)['previous_amendment_sha256'] == sha(amendment_path)
    amendment3_path = REG.parent / 'evidence_amendment_3.json'
    amendment3 = read(amendment3_path)
    assert sha(amendment3_path) == amendment3_path.with_suffix('.sha256').read_text().split()[0]
    assert amendment3['previous_amendment_sha256'] == sha(amendment2_path)
    for name, digest in amendment3['additional_inputs'].items():
        assert sha(ROOT / name) == digest
    historical_core = ROOT / 'artifacts/grm_c7/r2/fix4_continuation/before/core/graft_arena.py'
    # Prior art: Python AST structural comparison; local SHA-bound C7 source
    # archives (GRM, 2026). Compare exact implementations across FIX-4, not
    # whole-file identity. No prior art known to me for this audit composition.
    def fold_ast(path):
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'ArenaCache')
        funcs = {'consolidate', '_consolidation_prompts', '_format_step_prompt',
                 '_rare_tokens', '_caps_tokens', '_fact_set', '_coverage', '_digest_qc'}
        constants = {'DIGEST_PROMPTS', 'ERA_PROMPTS', 'CONSOLIDATE_NGEN', 'MIN_FOLD_KEEP',
                     'TEXT_SCAFFOLD_CONSOLIDATION', 'ALLOW_HIGH_COVERAGE_LIST_DIGESTS', '_FACT_STOP'}
        return [ast.dump(n, include_attributes=False) for n in cls.body if
                isinstance(n, ast.FunctionDef) and n.name in funcs or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    assert fold_ast(historical_core) == fold_ast(ROOT / 'core/graft_arena.py')
    for name, digest in reg['inputs'].items():
        assert sha(ROOT / name) == digest, name
    output.mkdir(parents=True, exist_ok=False)
    ns = pure_functions()
    A, harmony, score = ns['ArenaCache'], ns['harmony_turn'], ns['score']
    fixture = read(ROOT / 'artifacts/grm_c7/fixture.json')
    probes = {p['id']: p for p in fixture['probes']}
    base = ROOT / 'artifacts/grm_c7/r2'
    old = base / 'cells'
    new = base / 'fix4_attempt_1/cells'
    cells = sorted(old.glob('A-*/worker.json')) + sorted(new.glob('A-*/worker.json'))
    assert len(cells) == 39
    model_ids = set()
    for path in cells:
        w = read(path)
        assert read(path.with_name('controller.json'))['status'] == 'COMPLETE'
        model_ids.add((w['model_info']['repository'], w['model_info']['revision']))
    from tokenizers import Tokenizer
    tokenizer_path = Path(read(cells[-1])['model_info']['model_dir']) / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, historical = [], []
    for directory, target in [(new, rows), (old, historical)]:
        for path in sorted(directory.glob('*/probes.jsonl')):
            for line, text in enumerate(path.read_text().splitlines(), 1):
                row = json.loads(text)
                row['receipt'] = str(path.relative_to(ROOT)) + ':' + str(line)
                row['manifest'] = str((path.parent / 'checkpoint/repository/manifest.json').relative_to(ROOT))
                target.append(row)
    assert len(rows) == 52 and len(historical) == 8
    oracle, memory, proposals = [], [], []
    for row in rows:
        p = probes[row['probe_id']]
        for side in ['memory', 'oracle']:
            assert score(row[side]['answer'], p) == row[side]['score']
        o, m = row['oracle'], row['memory']
        assert o['source_texts'] == p['oracle_source_texts']
        assert o['wrapped_prompt'] == harmony(o['live_prompt'], None)
        assert tokenizer.encode(o['wrapped_prompt'], add_special_tokens=False).ids == o['prompt_token_ids']
        assert not o['mounted_ids']
        label = ('ANSWERABLE_ALIAS_ABSTENTION' if p['class'] == 'alias' else
                 'ANSWERABLE_DIRECT_ABSTENTION') if p['answerable'] else 'VALID_UNANSWERABLE_CONTROL_CASE_MISMATCH'
        if o['score']['abstained']:
            oracle.append({'probe_id': p['id'], 'class': p['class'], 'distance': p['distance'],
                'answerable': p['answerable'], 'classification': label,
                'source_present': True, 'asked_fact_present': bool(p['answerable']),
                'expected': p['expected'], 'served_text': o['answer'],
                'wrapped_prompt': o['wrapped_prompt'], 'prompt_token_ids': o['prompt_token_ids'],
                'receipt': row['receipt'], 'generated_token_count': 'NOT_RECORDED',
                'stop_reason': 'NOT_RECORDED', 'truncation_cause': 'NOT_ESTABLISHED'})
        ri = m['route_info']
        mounted = m['residency']['mounted_ids']
        trip_mounts = [t.get('mount_set', []) for t in ri.get('_route_observation', {}).get('trips', [])]
        if m['score']['abstained']:
            category = ('c_FIX4_recency' if ri.get('served_from') == 'recency_mount' else
                        'b_mounted_reader' if mounted else 'a_no_mount')
            manifest = read(ROOT / row['manifest'])
            mounted_texts = [{'node_id': i, 'text': manifest['nodes'][i]['text']} for i in mounted]
            memory.append({'probe_id': p['id'], 'class': p['class'], 'distance': p['distance'],
                'answerable': p['answerable'], 'abstention_error': m['score']['abstention_error'],
                'category': category, 'mounted_ids': mounted, 'trip_mounts': trip_mounts,
                'served_from': ri.get('served_from'), 'served_from_node_ids': ri.get('served_from_node_ids'),
                'admission_policy_branch': ri.get('admission_policy_branch'),
                'served_text': m['answer'], 'expected': p['expected'], 'receipt': row['receipt'],
                'manifest': row['manifest'], 'mounted_source_texts': mounted_texts})
        records = list(zip(p['source_turns'], p['oracle_source_texts']))
        rendered = latest_lines(records)
        for item in rendered:
            for provenance in item['provenance']:
                assert any(t == provenance['turn'] and provenance['quote'] in s for t, s in records)
        proposal = ('Source-grounded latest values and relations:\n' + '\n'.join(x['line'] for x in rendered) +
            '\n\nOriginal exact source records, in chronological order:\n' +
            '\n'.join(f'[turn {t}] {s}' for t, s in sorted(records)) +
            '\nUse only these records and relations. For a current value, use the latest assignment. '
            'An alias refers to the named entity. Do not infer unrecorded fields.\n' + row['question'])
        assert 'inspection password' not in '\n'.join(x['line'] for x in rendered)
        proposals.append({'probe_id': p['id'], 'question': row['question'], 'source_records': records,
            'rendered_lines': rendered, 'baseline_wrapped_prompt': o['wrapped_prompt'],
            'proposal_wrapped_prompt': harmony(proposal, None),
            'proposal_prompt_token_ids': tokenizer.encode(harmony(proposal, None), add_special_tokens=False).ids,
            'ngen': 32, 'stop_sequences': list(ns['HARMONY_STOPS']), 'model_result': 'NOT_RUN',
            'scope': 'oracle-only; frozen scorer and memory unchanged'})
    # Registered hand-case checks of provenance/chronology/negative controls.
    hand = latest_lines([(2, 'The current C7-Test-0 value is New-2.'),
                         (1, 'The current C7-Test-0 value is Old-1.'),
                         (3, 'C7-Link-0 is an alias for C7-Test-0.')])
    assert any('latest value = New-2' in x['line'] for x in hand)
    assert any('through C7-Test-0 = New-2' in x['line'] for x in hand)
    assert not any('Old-1' in x['line'] for x in hand)
    assert latest_lines([]) == []
    assert len(latest_lines([(1, 'C7-Link-2 is an alias for C7-Missing-2.')])) == 1
    folds = []
    for path in sorted(list(old.glob('*/folds/*/result.json')) + list(new.glob('*/folds/*/result.json'))):
        result, start = read(path), read(path.with_name('start.json'))
        worker = read(path.parents[2] / 'worker.json')
        source_hash = worker['executed_inputs'][str(ROOT / 'core/graft_arena.py')]
        expected_core = historical_core if old in path.parents else ROOT / 'core/graft_arena.py'
        assert source_hash == sha(expected_core) == start['binding']['fold_core_sha256']
        srcs = [x['text'] for x in start['sources']]
        need = A._fact_set(srcs)
        assert len(need) == result['result']['need_count']
        trace = [json.loads(line)['text'] for line in path.with_name('full_candidates.jsonl').read_text().splitlines()]
        parsed = trace_attempts(trace, ns['HARMONY_STOPS'])
        attempts = []
        # Prior art: local core's registered prompt primers and observer
        # receipts (GRM, 2026). Identify executed family by all three recorded
        # candidate prefixes; do not silently trust the observer's null-kind
        # deep test. Reconstruction only, no generation or prompt repair.
        matching = [prompts for prompts in (A.DIGEST_PROMPTS, A.ERA_PROMPTS)
                    if all(a['text'].startswith(p.rsplit('Assistant:', 1)[1].strip())
                           for a, p in zip(result['attempts'], prompts))]
        assert len(matching) == 1
        actual_prompts = matching[0]
        for a, tr, prompt in zip(result['attempts'], parsed, actual_prompts):
            primer = prompt.rsplit('Assistant:', 1)[1]
            decoded = tr['raw_final_decode']
            for stop in ns['HARMONY_STOPS']:
                decoded = decoded.split(stop, 1)[0]
            rebuilt = (primer + ' ' + decoded).strip()
            for stop in ('\nUser:', 'User:'):
                rebuilt = rebuilt.split(stop, 1)[0]
            assert rebuilt.strip() == a['text'], (start['turn'], a['prompt_index'], repr(rebuilt.strip()), repr(a['text']))
            qc = A._digest_qc(a['text'], None, forbid_lists=True)
            raw_coverage = A._coverage(a['text'], need)
            assert qc == a['qc']
            assert (raw_coverage if qc else 0) == a['coverage']
            have = A._rare_tokens(a['text']) | A._caps_tokens(a['text'], False)
            user_text = prompt.removeprefix('User: ').rsplit('\nAssistant:', 1)[0]
            attempts.append({**a, **tr, 'lexical_coverage_before_QC': raw_coverage,
                'covered_facts': sorted(need & have), 'missing_facts': sorted(need - have),
                'wrapped_prompt_reconstruction': harmony(user_text, None) + primer,
                'prompt_evidence_class': 'source-bound reconstruction; runtime input token IDs not recorded'})
        best = max((a for a in attempts if a['qc']), key=lambda a: a['coverage'])
        assert best['coverage'] == result['result']['best_cov'] < 0.70
        assert not result['accepted'] and result['digest_text'] is None
        folds.append({'turn': start['turn'], 'origin': 'inherited_pre_FIX4' if old in path.parents else 'FIX4_continuation',
            'receipt': str(path.relative_to(ROOT)), 'sources': start['sources'], 'need': sorted(need),
            'need_count': len(need), 'required_hits': math.ceil(.7 * len(need)),
            'best_hit_count': best['hit_count'], 'best_coverage': best['coverage'],
            'best_missing_facts': best['missing_facts'], 'best_text': best['text'],
            'all_attempts': attempts, 'digest_text': None, 'accepted': False,
            'FIX3_path': 'YES: bound executed source, configured Harmony oracle, recorded stop short-circuit',
            'executed_arena_sha256': source_hash,
            'recorded_start_prompts': start['prompts'],
            'start_prompt_mismatch': list(actual_prompts) != start['prompts'],
            'binding_constraint': 'coverage below 0.70; budget hits are additional candidate-level failures'})
    folds.sort(key=lambda x: x['turn'])
    assert len(folds) == 13
    summary = {'evidence_class': 'CPU replay of lead-run model receipts; no model execution',
        'cells': {'preserved': 3, 'continuation': 36, 'total': 39},
        'probe_rows': len(rows), 'historical_probe_rows_excluded': len(historical),
        'score_totals': {s: {k: sum(r[s]['score'][k] for r in rows) for k in
                           ['exact_correct', 'exact_answer_error', 'abstained', 'abstention_error',
                            'wrong_value_error', 'unsupported_answer_error']} for s in ['memory', 'oracle']},
        'oracle_classes': dict(Counter(r['classification'] for r in oracle)),
        'memory_answerable_split': dict(Counter(r['category'] for r in memory if r['answerable'])),
        'memory_all_abstained_split': dict(Counter(r['category'] for r in memory)),
        'memory_by_class': {c: dict(Counter(r['category'] for r in memory if r['class'] == c and r['answerable']))
                           for c in ['fresh', 'alias', 'correction', 'folded']},
        'folds': len(folds), 'fold_candidates': sum(len(f['all_attempts']) for f in folds),
        'fold_budget_exhausted': sum(a['budget_exhausted'] for f in folds for a in f['all_attempts']),
        'fold_stopped_early': sum(not a['budget_exhausted'] for f in folds for a in f['all_attempts']),
        'fold_QC_rejections': sum(not a['qc'] for f in folds for a in f['all_attempts']),
        'start_prompt_mismatch_turns': [f['turn'] for f in folds if f['start_prompt_mismatch']],
        'memory_beats_oracle_rows': [r['probe_id'] for r in rows if r['memory']['score']['exact_correct'] and
                                     not r['oracle']['score']['exact_correct']],
        'max_continuation_residency': max(json.loads(l)['summed_token_seats'] for p in new.glob('*/residency.jsonl')
                                         for l in p.read_text().splitlines()),
        'model_ids': sorted(model_ids), 'tokenizer_sha256': sha(tokenizer_path),
        'proposal_model_quality': 'NOT_RUN', 'CPU_replay_gates': 'PASS',
        'core_fold_ruling': 'STOP_AND_REPORT_SCOUT_FIX_5_CANDIDATE; not claimed fixed',
        'registration_sha256': sha(REG)}
    for name, value in [('summary.json', summary), ('oracle_abstentions.json', oracle),
                        ('memory_abstentions.json', memory), ('folds.json', folds),
                        ('oracle_proposal_prompts.json', proposals)]:
        dump(output / name, value)
    oracle_md = ['# Oracle abstentions: all 34 continuation rows',
        'CPU receipt replay. Sixteen answerable failures; eighteen valid negative controls with exact-case mismatch. '
        'Every wrapped prompt and served text below is quoted verbatim from the named JSONL receipt.',
        table(['Probe', 'Class', 'Classification', 'Expected', 'Served'],
              [[r['probe_id'], r['class'], r['classification'], r['expected'], r['served_text']] for r in oracle])]
    for r in oracle:
        oracle_md += [f"## {r['probe_id']}", f"Receipt: `{r['receipt']}`. Classification: {r['classification']}.",
                      'Wrapped prompt (verbatim):\n```text\n' + r['wrapped_prompt'] + '\n```',
                      'Served text (verbatim):\n```text\n' + r['served_text'] + '\n```']
    (output / 'ORACLE_TABLE.md').write_text('\n\n'.join(oracle_md) + '\n')
    fold_md = ['# Fold table: all 13 rejected events',
        'CPU replay of source-bound receipts. No accepted digest exists; candidate texts below are verbatim. '
        'Token counts are reconstructed from the exact generation loop and decode-call trace, not retokenized estimates.',
        table(['Turn', 'Origin', 'Best hits / need', 'Required hits', 'Coverage', 'Tokens p0/p1/p2', 'Cap-hit prompts'],
              [[f['turn'], f['origin'], f"{f['best_hit_count']}/{f['need_count']}", f['required_hits'],
                f['best_coverage'], '/'.join(str(a['generated_tokens_from_decode_trace']) for a in f['all_attempts']),
                ','.join(str(a['prompt_index']) for a in f['all_attempts'] if a['budget_exhausted']) or 'none'] for f in folds])]
    for f in folds:
        fold_md += [f"## Turn {f['turn']}", f"Receipt: `{f['receipt']}`. FIX-3: {f['FIX3_path']}. "
                    f"Need {f['need_count']}; required {f['required_hits']}; best coverage {f['best_coverage']}. "
                    'All three candidates rejected; digest_text is null.']
        for s in f['sources']:
            fold_md += [f"Source node {s['id']} ({s['ntok']} tokens):\n```text\n{s['text']}\n```"]
        for a in f['all_attempts']:
            fold_md += [f"### Candidate {a['prompt_index']}",
                f"QC={a['qc']}; recorded coverage={a['coverage']}; lexical coverage before QC={a['lexical_coverage_before_QC']}; "
                f"generated tokens={a['generated_tokens_from_decode_trace']}; stop={a['stop']}; cap hit={a['budget_exhausted']}.",
                '```text\n' + a['text'] + '\n```', 'Missing lexical facts: ' + ', '.join(a['missing_facts']) + '.',
                'Wrapped prompt (source-bound reconstruction, not a recorded input):\n```text\n' +
                a['wrapped_prompt_reconstruction'] + '\n```']
        if f['start_prompt_mismatch']:
            fold_md += ['Observer discrepancy: start.json reports ERA prompts because kind=null is treated as deep. '
                        'Actual candidate prefixes match source-bound DIGEST prompts. The reconstruction above uses '
                        'that matching family. Recorded start prompts (verbatim):\n```text\n' +
                        '\n\n'.join(f['recorded_start_prompts']) + '\n```']
    (output / 'FOLD_TABLE.md').write_text('\n\n'.join(fold_md) + '\n')
    (output / 'MEMORY_TABLE.md').write_text('# Memory abstentions\n\n'
        'All 48 abstaining rows are retained; answerable=yes selects the requested 30 errors. '
        'Category c takes precedence over b because recency also mounts. '
        'Actual residency and per-trip mount sets are authoritative; nominal route_info.mounts is not used.\n\n' +
        table(['Probe', 'Class', 'Answerable', 'Category', 'Actual mounts', 'Served text'],
              [[r['probe_id'], r['class'], r['answerable'], r['category'], r['mounted_ids'], r['served_text']] for r in memory]) + '\n')
    with (output / 'MEMORY_TABLE.md').open('a') as stream:
        for r in memory:
            stream.write(f"\n## {r['probe_id']}\n\nReceipt: `{r['receipt']}`; mounted text from `{r['manifest']}`.\n")
            for s in r['mounted_source_texts']:
                stream.write(f"\nNode {s['node_id']}:\n```text\n{s['text']}\n```\n")
    # Historical rows are quoted separately and never added to the 52-row denominator.
    with (output / 'HISTORICAL_ORACLE_QUARANTINE.md').open('w') as stream:
        stream.write('# Historical oracle quarantine\n\nEight original rows, NOT_MEASURED under the FIX-4 continuation policy. '
                     'Quoted for completeness; excluded from every continuation metric and proposal.\n')
        for r in historical:
            stream.write(f"\n## {r['probe_id']}\n\nReceipt: `{r['receipt']}`.\n\nWrapped prompt:\n```text\n" +
                         r['oracle']['wrapped_prompt'] + '\n```\n\nServed text:\n```text\n' + r['oracle']['answer'] + '\n```\n')
    for name, digest in reg['inputs'].items():
        assert sha(ROOT / name) == digest, name
    for extra in [amendment, amendment3]:
        for name, digest in extra['additional_inputs'].items():
            assert sha(ROOT / name) == digest, name
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory; no overwrite')
    run(parser.parse_args().output)
