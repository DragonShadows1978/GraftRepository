#!/usr/bin/env python3
"""Isolated r2 registration and reporting, using unchanged X3 fork primitives.
Prior art: house X3/DET1 (2026) content hashes, fixed classifier and contingency
counts. Amendment 2 supplies new strata quotas and prior-run sham floor. No new
statistical estimator; no prior art known to me for these particular thresholds.
"""
import copy
import unicodedata
from scripts.grm_x3_diagnostic import *
from scripts import grm_x3_diagnostic as r1
from scripts.grm_x3_r2_scorer import outcome
from scripts.grm_x3_r2_anchor import REGISTRATION_SHA256, MANIFEST_SHA256
OUT=ROOT/'artifacts/grm_x3/r2'


def validate_fixtures(out=None):
    out=Path(out) if out is not None else OUT
    if sha(out/'registration.json')!=REGISTRATION_SHA256:
        raise X3Error('forged/stale r2 registration')
    if sha(out/'fixtures/manifest.json')!=MANIFEST_SHA256:
        raise X3Error('forged/stale r2 fixture manifest')
    reg=json.loads((out/'registration.json').read_text())
    manifest=json.loads((out/'fixtures/manifest.json').read_text())
    records=[reg['order'],reg['parent_registration'],reg['r1_evidence'],reg['tokenizer'],
             reg['scorer']['source'],reg['scorer']['controls'],*manifest['fixtures'],*manifest['source_pins']]
    for rec in records:
        if sha(ROOT/rec['path'])!=rec['sha256']:raise X3Error('frozen r2 input drift: '+rec['path'])
    if unicodedata.unidata_version!=reg['scorer']['unicode_version']:raise X3Error('Unicode version drift')
    rows=[json.loads((ROOT/rec['path']).read_text()) for rec in manifest['fixtures']]
    if [r['id'] for r in rows]!=[f'x3_r2_{i:02d}' for i in range(20)]:raise X3Error('r2 fixture allocation drift')
    if {g:sum(r['intended_group']==g for r in rows) for g in ('correct','decoy','refusal')}!=manifest['counts']:
        raise X3Error('r2 strata count drift')
    for r in rows:
        if len(r['base_token_ids'])!=len(r['swap_token_ids']) or not 0<len(r['base_token_ids'])+len(r['sham_token_ids'])<=96:
            raise X3Error('r2 geometry drift')
    return reg,rows


def summarize(rows,reg):
    # Reuse r1's descriptive arithmetic on r2 rows only; historical r1 receipts
    # are never opened here. Remap IDs privately for its fixed 20-row validator.
    if any(r['id'] not in {f'x3_r2_{i:02d}' for i in range(20)} for r in rows):raise X3Error('non-r2 result')
    mapped=copy.deepcopy(rows)
    for r in mapped:r['id']=r['id'].replace('x3_r2_','x3_')
    report=r1.summarize(mapped,reg)
    full=len(rows)==20 and len({r['id'] for r in rows})==20
    realized=[r for r in rows if r['outcome']['category_realized'] and not r['outcome']['truncated']]
    strata={g:sum(r['intended_group']==g for r in realized) for g in reg['strata_minimum']}
    valid=full and not any(r['outcome']['truncated'] for r in rows) and all(strata[g]>=n for g,n in reg['strata_minimum'].items())
    correct=[r for r in realized if r['intended_group']=='correct']
    decoy=[r for r in realized if r['intended_group']=='decoy']
    threshold=reg['scoring']['mass_threshold']
    def accuracy(items):
        return {w:sum(((r['mass']<threshold) if w=='mass' else (r['forks']['remove']['kl_nats']>1))==r['outcome']['exact_error'] for r in items)/len(items) if items else None for w in ('mass','lesion')}
    acc=accuracy(realized);dacc=accuracy(decoy)
    high=sum(r['forks']['remove']['kl_nats']>1 for r in correct)
    dhigh=sum(r['mass']>=threshold for r in decoy)
    sham_good=sum(r['forks']['sham']['kl_nats']<reg['sham_calibration']['threshold_nats'] for r in rows)
    p1=report['same_payload_equal_count']==20 and sham_good>=18 if full else None
    predictions={"P1'":p1,"P2'":high>=6 if valid else None,
        "P3'":dhigh>=4 and dacc['lesion']>=dacc['mass'] if valid else None,
        "Q1'":p1,"Q2'":high<=5 if valid else None,"Q3'":acc['lesion']<=acc['mass'] if valid else None}
    # Unchanged kill rule uses overall20 frozen exact-error classifier accuracy.
    killed=full and (report['controls_comparable_count']>=3 or report['accuracy']['all']['lesion']<=report['accuracy']['all']['mass'])
    for r in mapped:r['outcome']['exact_error']=r['outcome']['value_span_error']
    span_table=r1.summarize(mapped,reg)['table']
    for cell in span_table:cell['value_span_errors']=cell.pop('exact_errors')
    report.update(schema='grm.x3.summary.r2.v1',
        status='RED_KILL' if killed else 'RED_STRATA_OR_INCOMPLETE' if not valid else 'RED_P1' if not p1 else 'COMPLETE',
        predictions=predictions,realized_strata=valid,realization_scorer='registered value-span',
        strata={g:{'intended':sum(r['intended_group']==g for r in rows),'realized_untruncated':strata[g],
                   'minimum':reg['strata_minimum'][g]} for g in strata},
        frozen_exact_equality_strata={g:sum(r['intended_group']==g and r['outcome']['exact_equality_realization'] and not r['outcome']['truncated'] for r in rows) for g in strata},
        accuracy_realized={'all':acc,'decoy':dacc},accuracy_label='frozen whole-answer exact-error; accuracy field is descriptive all20/intended-decoy',
        value_span_error_table=span_table,P2_high_count=high,decoy_mass_high_count=dhigh,
        sham_below_0_10_count=sham_good,sham_floor_statement=reg['sham_calibration']['statement'])
    report.pop('P1_joint_count') # r1's .05 joint rule is not an r2 guard.
    report['controls']['sham']['kl_below_0_10_count']=sham_good
    return report
