#!/usr/bin/env python3
"""Join XM1 receipts without allowing CPU evidence to unlock GPU claims.

Prior art: GRM RS4 residual_table/mean_over_positions and exact barriers
(GRM contributors, 2026). Reuse fed-text minus mount mass and the lead's
registered >0.10 / two-refusal falsifiers. Ours: cross-model completeness and
provenance joins. No new selection or attention algorithm.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import grm_xm1_parity as xm
from scripts.grm_xm1_registration import read,create,sha


def summarize(reg,directory,mode):
    gate=xm.barrier(directory,reg)
    rows=[]
    models=[]
    for model in reg['model_order']:
        paired=[]
        statuses=[]
        for probe in reg['probes']:
            cells={a:next(c for c in reg['cells'] if c['model']==model and c['probe']==probe and c['arm']==a) for a in reg['arms']}
            receipts={a:xm.verify_receipt(xm.cell_path(directory,c,mode),c,mode) if xm.cell_path(directory,c,mode).exists() else None for a,c in cells.items()}
            row=dict(model=model,probe=probe,role=reg['probes'][probe]['role'],arms={},fed_minus_mount=None,
                     historical_source_texts_equal=reg['probes'][probe]['capture_texts']==reg['probes'][probe]['feed_texts'])
            for arm,r in receipts.items():
                status=r['status'] if r else 'MISSING'
                statuses.append(status)
                result=(r or {}).get('result',{})
                full=result.get('per_layer',{}).get('full_attention',{}).get('mean_over_answer_positions',{})
                row['arms'][arm]=dict(status=status,served=result.get('served'),value_span=result.get('value_span'),
                    mass=full.get('fed_text_mass' if arm=='C5' else 'mounted_mass'),
                    receipt=str(xm.cell_path(directory,cells[arm],mode).relative_to(directory)))
            if all(row['arms'][a]['status']=='PASS' and row['arms'][a]['mass'] is not None for a in reg['arms']):
                row['fed_minus_mount']=row['arms']['C5']['mass']-row['arms']['C3l']['mass']
                row['gap_exceeds_0_10']=row['fed_minus_mount']>0.10
                row['correct_fed_to_refusal_mount']=bool(row['arms']['C5']['value_span']['correct'] and row['arms']['C3l']['value_span']['refusal'])
                paired.append(row)
            rows.append(row)
        complete=len(paired)==len(reg['probes'])
        falsified=any(r['gap_exceeds_0_10'] for r in paired) or sum(r['correct_fed_to_refusal_mount'] for r in paired)>=2
        models.append(dict(model=model,pairs=len(paired),statuses=statuses,
            verdict='CPU_DOUBLE_ONLY' if mode=='cpu' else
                    'BLOCKED_REFERENCE' if gate['status']!='PASS' else
                    'NO_GRAFT_PATH' if set(statuses)=={'NO_GRAFT_PATH'} else
                    'NON_FIT' if 'NON_FIT' in statuses else
                    'INCOMPLETE' if not complete else
                    'FALSIFIER_TRIGGERED' if falsified else 'NOT_FALSIFIED_ON_PANEL',
            limits='Historical source-text contrast; not same-token equivalence. Refusal detector lexical; inspect raw served answers.'))
    return dict(schema='grm.xm1.results.v1',mode=mode,registration_sha256=sha(xm.REG),
                reference_barrier=gate,models=models,rows=rows,
                evidence_class='CPU numerical/native-source double only' if mode=='cpu' else 'GPU measured cells',
                answer_to_david='UNRESOLVED: no same-token cross-model causal claim follows from the historical RS4 source sets.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['cpu','gpu'],required=True)
    p.add_argument('--directory',type=Path,default=xm.OUT)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    create(a.output,summarize(xm.validate_registration(),a.directory,a.mode))
