#!/usr/bin/env python3
"""Registered XM2 Qwen worker, dry-run by default; never an outer flock.

Prior art: GRM XM1/LT1 immutable source/receipt bindings and CMC1 fail-fast
GPU lease (2026). Reused without scheduling novelty. Ours: fixed 16-cell
reservation budget, original reference barrier, fail-closed final completion.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import grm_xm1_parity as xm
from scripts.grm_xm1_parity import gpu_loader
from scripts.grm_xm1_registration import read,sha,create
OUT=ROOT/'artifacts/grm_xm2'
REG=OUT/'registration.json'
AMENDMENT=OUT/'registration_amendment_1.json'
RUN=OUT/'amendment_1_run'
REFERENCE=ROOT/'artifacts/grm_xm1/amendment_1_run'


def validate():
    parent=xm.validate_registration()
    reg=xm.xm2_registration()
    if reg['xm1_registration_sha256']!=sha(xm.REG):
        raise xm.XM1Error('XM2 parent registration mismatch')
    if (reg['prior_reserved_s'] + sum(c['reservation_s'] for c in reg['cells'])
            > reg['gpu_budget_s']):
        raise xm.XM1Error('XM2 budget exceeds 0.5 GPU-h')
    if os.environ.get('TC_WEIGHT_BITS','4') != '4':
        raise xm.XM1Error('registered XM1 Qwen loader requires TC_WEIGHT_BITS=4')
    gate=xm.barrier(REFERENCE,parent)
    if gate['status']!='PASS':
        raise xm.XM1Error('STOP: original GPT-OSS barrier '+json.dumps(gate))
    return reg,parent


def receipt_path(cell):
    return RUN/'cells'/f"{cell['id']}.json"


def run(cell,reg,parent):
    binding=dict(registration_sha256=sha(REG),amendment_sha256=sha(AMENDMENT),cell=cell)
    path=receipt_path(cell)
    if path.exists():
        receipt=read(path)
        if receipt['binding']!=binding: raise xm.XM1Error('resume binding mismatch')
        return receipt
    # Prior art: CMC1 lease + XM1 pessimistic lost-worker accounting (2026).
    # Charge the full fixed reservation before loading, including failed or
    # interrupted attempts; one attempt per cell. No automatic retry.
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    attempts=RUN/'attempts'
    attempts.mkdir(parents=True,exist_ok=True)
    with gpu_lease(cell['reservation_s'],0):
        prior_paths=sorted((OUT/'run'/'attempts').glob('*.json'))
        if {str(p.relative_to(ROOT)):sha(p) for p in prior_paths} != reg['prior_attempt_pins']:
            raise xm.XM1Error('STOP: original XM2 attempt set changed; amendment required')
        used=reg['prior_reserved_s']+sum(read(p)['reservation_s'] for p in attempts.glob('*.json'))
        if used+cell['reservation_s']>reg['gpu_budget_s']:
            raise xm.XM1Error('STOP: XM2 budget exhausted')
        prior=prior_paths+sorted(attempts.glob('*.json'))
        if prior:
            last=max(read(p)['started_unix_s']+read(p)['reservation_s'] for p in prior)
            if time.time()<last+30:
                raise xm.XM1Error('STOP: registered 30 s gap not yet satisfied; no wait')
        create(attempts/f"{cell['id']}.json",dict(reservation_s=cell['reservation_s'],
          started_unix_s=time.time(),binding=binding))
        start=time.monotonic()
        receipt=dict(binding=binding,status='ERROR',gpu_executed=False,
          evidence_class='GPU Qwen native cell; historical source contrast')
        loaded=None
        try:
            with xm.legacy_environment(parent):
                os.environ['GRM_QWEN35_FINAL_CHANNEL']='1'
                receipt['gpu_executed']=True
                # Prior art: XM1 native loader (GRM contributors, 2026).
                # Import/call the identical callable, with no adapter override.
                # CPU doubles exercise this call and XM1 execute's default call.
                loaded=gpu_loader('qwen35',parent)
                if loaded.info.get('revision')!=reg['model_revision']:
                    raise xm.XM1Error('Qwen model revision mismatch')
                if loaded.info.get('weight_bits')!=4:
                    raise xm.XM1Error('Qwen weight mode differs from XM1 INT4 receipts')
                # Check cached tokenizer/template content against registered
                # local copies before any cell forward.
                model_dir=Path(loaded.info['model_dir'])
                for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
                    if sha(model_dir/name)!=sha(OUT/'sources'/name):
                        raise xm.XM1Error('Qwen tokenizer/template mismatch: '+name)
                result=xm.native_cell(loaded,cell,parent)
                receipt['result']=result
                receipt['status']='PASS' if result['decode_status']=='FINAL' else result['decode_status']
        except Exception as exc:
            receipt.update(status='ERROR',error_type=type(exc).__name__,error=str(exc))
        finally:
            if loaded is not None:
                for _,att in loaded.attentions():
                    att.inject_kv,att.graft_seats,att.live_shift=None,0,None
            receipt['elapsed_s']=time.monotonic()-start
            create(path,receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',required=True)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    reg,parent=validate()
    cell=next((c for c in reg['cells'] if c['id']==args.cell),None)
    if cell is None: parser.error('unregistered cell')
    if args.dry_run or not args.run:
        print(json.dumps(dict(status='DRY_RUN',cell=cell,gpu_executed=False,
          reference_barrier='PASS',source_pins='PASS',total_reserved_s=sum(c['reservation_s'] for c in reg['cells']),
          worker_leases_itself=True)))
        return 0
    result=run(cell,reg,parent)
    print(json.dumps(dict(status=result['status'],cell=cell['id'])))
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':
    raise SystemExit(main())
