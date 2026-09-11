"""r4 authority and receipt diagnostics, CPU-only.

Prior art: house X1 (2026), locally verified immutable continuations and paired
query definitions; reused without new scientific algorithm. New: lead-selected
subset and historical-unit adoption. NIST FIPS 180-4 (2015) SHA-256,
unverified — lead to check: NIST FIPS 180-4 SHA-256.
"""
from __future__ import annotations

import ast
import re
import unicodedata

from scripts import grm_x1_campaign as c
from scripts.grm_x1_register import create, raw_json, sha

SOURCE_AMENDMENT_REG_SHA = '98aceb90e852ca5c1de70f94703fdcda662ed9be08e92fc150eae00bc7126269'
REG_SHA = '288ba57b95fd5bb88e6fa9660fcb4733cddb648867950986bfe6c23d3602b420'


def payload():
    from scripts import grm_x1_units as u
    path = c.OUT / 'continuation_04_registration.json'
    if sha(path) != REG_SHA:
        raise ValueError('r4 registration binding mismatch')
    reg = c.read(path)
    bindings = {**reg['historical_files'],
                'artifacts/grm_x1/continuation_04_registration.json': REG_SHA,
                'orders/GRM_X1_AMENDMENT_4.md': reg['order_sha256']}
    for name, digest in bindings.items():
        if sha(c.ROOT / name) != digest:
            raise ValueError(f'r4 historical/authority binding drift: {name}')
    sources = c.source_manifest()
    before, allowed = reg['source_shas_before'], set(reg['allowed_sources'])
    if (set(before) - set(sources) or set(sources) - set(before) - allowed
            or any(sources[p] != d for p, d in before.items() if p not in allowed)):
        raise ValueError('r4 source drift outside harness scope')
    # House X1 (2026) byte-preserved reader/scoring contract. Only aggregation
    # population and receipt diagnostics change; keep scientific unit identical.
    source = (c.ROOT / 'scripts/grm_x1_campaign.py').read_text()
    scoring = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name in ('score', 'normalize_answer')}
    scoring['REFUSALS'] = sorted(c.REFUSALS)
    if scoring != reg['scoring_source']:
        raise ValueError('r4 scoring definition drift')
    layout = [x for x in u.units(full=True) if x['cell'] in reg['cells']]
    prior = c.read(c.OUT / 'continuation_03.json')
    if raw_json(layout) != raw_json(reg['units']) or any(
            raw_json(unit) != raw_json(next(x for x in prior['units'] if x['unit_id'] == unit['unit_id']))
            for unit in layout):
        raise ValueError('r4 unit definition not byte-identical to r3')
    reuse = reg['reuse']
    receipt = c.read(c.ROOT / reuse['receipt_path'])
    unit = next(x for x in layout if x['unit_id'] == reuse['unit_id'])
    if (u.rows_digest(unit) != reuse['unit_definition_sha256']
            or receipt['status'] != 'COMPLETE'
            or receipt['fingerprint'] != reg['predecessor_sha256']
            or receipt['worker_wall_s'] != reuse['worker_wall_s']):
        raise ValueError('r4 historical unit reuse binding mismatch')
    u.check_rows(unit, receipt['rows'], True)
    return {'schema': 'grm.x1.continuation.v4', 'bindings': bindings,
            'sources': sources, 'lead_commands_sha256': sha(c.OUT / 'lead_commands.txt'),
            **{k: reg[k] for k in ('cells', 'unit_scheme', 'budget', 'rails', 'failure_policy',
                                   'historical_files', 'not_run', 'oracle_positive', 'natural_unlock',
                                   'served_text_class')},
            'units': layout, 'reuse': {**reuse, 'status': 'VALID_BYTE_IDENTICAL'},
            'receipt_directory': 'artifacts/grm_x1/receipts/r4',
            'claim_directory': 'artifacts/grm_x1/claims/r4',
            'evidence_class': 'reasoning: sealed before CPU gates; r3 receipt reused, no new GPU run'}


def seal():
    c.verify_fixtures()
    value = payload()
    path, checksum = c.OUT / 'continuation_04.json', c.OUT / 'continuation_04.sha256'
    if path.exists() or checksum.exists():
        raise FileExistsError('r4 continuation immutable')
    create(path, raw_json(value))
    create(checksum, f'{sha(path)}  continuation_04.json\n'.encode())
    return {'continuation': str(path), 'sha256': sha(path)}


def validate():
    path = c.OUT / 'continuation_04.json'
    if sha(path) != (c.OUT / 'continuation_04.sha256').read_text().split()[0]:
        raise ValueError('r4 continuation SHA mismatch')
    value = c.read(path)
    expected = payload()
    supplement = c.OUT / 'continuation_04_source_amendment_01.json'
    supplement_sha = None
    if supplement.exists():
        if sha(supplement) != supplement.with_suffix('.sha256').read_text().split()[0]:
            raise ValueError('r4 source amendment SHA mismatch')
        amendment = c.read(supplement)
        if amendment != source_amendment_payload():
            raise ValueError('r4 source amendment binding mismatch (forged or stale)')
        # House X1 (2026): source-only correction in a separate immutable seal.
        # Compare every original policy field; only source hashes are superseded.
        expected['sources'] = value['sources']
        supplement_sha = sha(supplement)
    if value != expected:
        raise ValueError('r4 continuation policy/source binding mismatch (forged or stale)')
    c.verify_fixtures()
    if supplement_sha:
        return {**value, 'sources': amendment['sources'], 'source_amendment_sha256': supplement_sha}
    return value


def source_amendment_payload():
    path = c.OUT / 'continuation_04_source_amendment_01_registration.json'
    if sha(path) != SOURCE_AMENDMENT_REG_SHA:
        raise ValueError('r4 source amendment registration binding mismatch')
    reg = c.read(path)
    parent = c.OUT / 'continuation_04.json'
    if sha(parent) != reg['parent_sha256'] or sha(c.OUT / 'continuation_04_registration.json') != reg['registration_sha256']:
        raise ValueError('r4 source amendment parent binding mismatch')
    before, sources = c.read(parent)['sources'], c.source_manifest()
    allowed = set(reg['allowed_sources'])
    if set(before) != set(sources) or any(sources[p] != d for p,d in before.items() if p not in allowed):
        raise ValueError('r4 source amendment source scope drift')
    return {'schema': 'grm.x1.source-amendment.v1', 'parent_sha256': reg['parent_sha256'],
            'registration_sha256': SOURCE_AMENDMENT_REG_SHA, 'sources': sources,
            'evidence_class': 'reasoning: separate source correction seal; original r4 continuation untouched'}


def served_text_class(row):
    # Prior art: house X1 (2026) existing scored abstention flag reused first.
    # No prior art known to me for this exact anchored refusal template.
    # New descriptive, intentionally limited label; never changes any score.
    if row['abstained']:
        return 'abstention'
    text = unicodedata.normalize('NFKC', row['answer']).casefold().replace('’', "'")
    text = ' '.join(text.split()).strip()
    pattern = (r"(?:i'm sorry[, ]*(?:but )?|sorry[, ]*(?:but )?)?"
               r"i (?:can't|cannot|can not|am unable to) (?:help|assist|comply|provide)"
               r"(?: with (?:that|this|(?:that|this|your|the) request)| (?:assistance|information))?[.!]?")
    return 'refusal-style' if re.fullmatch(pattern, text) else 'answer'


def classify_rows(rows):
    return [{**row, 'served_text_class': served_text_class(row)} for row in rows]


def text_counts(rows):
    return {label: sum(served_text_class(r) == label for r in rows)
            for label in ('answer', 'abstention', 'refusal-style')}
