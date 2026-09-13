#!/usr/bin/env python3
"""Bounded, isolated XM1 runtime mutants after the passing author baseline.

Prior art: mutation testing, DeMillo/Lipton/Sayward (1978), unverified — lead
to check: Hints on Test Data Selection Help for the Practicing Programmer.
Taken: inject defects and require tests to reject them. Ours: five harness
seam faults, each in a fresh foreground Python process, no source edits.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.grm_xm1_registration import OUT,read,create,sha

FAULTS={
 'float_tolerance': '''old=x.reference_diff
def changed(a,b):
    import math
    return {k:v for k,v in old(a,b).items() if not (isinstance(v['expected'],float) and isinstance(v['observed'],float) and math.isclose(v['expected'],v['observed'],rel_tol=1e-8,abs_tol=1e-8))}
x.reference_diff=changed
''',
 'served_not_checked': '''old=x.reference_diff
def changed(a,b):
    result=old(a,b);result.pop('served',None);return result
x.reference_diff=changed
''',
 'capture_pin_ignored': '''old=x.capture
def changed(loaded,ids,shift):return old(loaded,ids,0)
x.capture=changed
''',
 'value_rotated': '''old=x.seat
def changed(loaded,payload,width):
    result=old(loaded,payload,width)
    for _,att in loaded.attentions():
        value=att.inject_kv[0 if loaded.kind=='mla' else 1]
        value.a[...,0]+=1
    return result
x.seat=changed
''',
 'partition_extra_mass': '''old=x.split_full_row
def changed(values,bounds):
    row=old(values,bounds);row['mounted_mass']+=0.1;return row
x.split_full_row=changed
''',
}


def run():
    registered=read(OUT/'mutation_registration.json')
    if registered['script_sha256']!=sha(Path(__file__)) or registered['faults']!=list(FAULTS):
        raise RuntimeError('mutation registration drift')
    results=[]
    for name,body in FAULTS.items():
        with tempfile.TemporaryDirectory(dir=OUT/'tmp',prefix='mutation_') as tmp:
            # Numerical/loader boundaries only; source modules remain untouched.
            code="from scripts import grm_xm1_parity as x\n"+body+"\nimport pytest\nraise SystemExit(pytest.main(['-q','--basetemp',"+repr(str(Path(tmp)/'pytest'))+",'tests/test_grm_xm1_parity.py','-k','not lead_commands']))\n"
            p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True,timeout=45)
            # Pytest 1 = assertion/test failure; 2+ is an ERROR mutant, never
            # counted as a kill. A surviving mutant is a coverage failure.
            results.append(dict(mutant=name,returncode=p.returncode,killed=p.returncode==1,
                error=p.returncode not in (0,1),stdout=p.stdout,stderr=p.stderr))
    eligible=[r for r in results if not r['error']]
    fraction=sum(r['killed'] for r in eligible)/len(eligible) if eligible else 0.
    payload=dict(evidence_class='author runtime-seam mutation test; not blind red team',results=results,
        killed_fraction=fraction,threshold=registered['threshold'],
        verdict='PASS' if len(eligible)==len(FAULTS) and fraction>=registered['threshold'] else 'RED',
        survivors=[r['mutant'] for r in results if not r['killed']],sources_edited=False)
    create(OUT/'mutation_results.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='results'}))

if __name__=='__main__':run()
