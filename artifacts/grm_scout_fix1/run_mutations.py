"""Registered fault reintroduction in disposable source overlays only.
Prior art: mutation testing, DeMillo/Lipton/Sayward (1978),
Hints on Test Data Selection, DOI 10.1109/C-M.1978.218136. We reuse the method;
only these five regression-specific mutations are ours. No novelty claimed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/grm_scout_fix1'
cases = [
    ('b1_flush_guard', 'core/grm_demand.py',
     'observer.early_abort and index < observer.ngen', 'observer.early_abort',
     'tests/test_grm_scout_fix1_demand.py::test_final_flush_only_fire_keeps_completed_answer'),
    ('b2_admission_prefix', 'core/grm_three_pass.py',
     '"live_segments_",\n                               "admission_")', '"live_segments_")',
     'tests/test_grm_scout_fix1_receipt.py::test_rt1_receipt_projection'),
    ('b3_serialize', 'core/graft_repository.py',
     '**({"capture": capture} if capture else {})', '**{}',
     'tests/test_grm_scout_fix1_manifest.py::test_capture_save_reload_round_trip'),
    ('b3_reload', 'core/graft_repository.py',
     'g.update(n.get("capture", {}))', 'g.update({})',
     'tests/test_grm_scout_fix1_manifest.py::test_capture_save_reload_round_trip'),
    ('b5_sum', 'scripts/grm_wc1_results.py',
     'return None if value is None else int(value)',
     'return None if value is None else len(info.get("mount_fitted") or [])',
     'tests/test_grm_scout_fix1_wc1.py::test_wc1_token_seats_and_resume_dedup'),
]
source_paths = sorted({p for _,p,*_ in cases})
hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
results = []
for name,path,old,new,test in cases:
    source = (ROOT/path).read_text()
    assert source.count(old) == 1, name
    directory = OUT/'mutants'/name
    target = directory/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.replace(old,new))
    log = OUT/f'mutation_{name}.log'
    command = [sys.executable,str(OUT/'run_snapshot_tests.py'),str(directory),'-q',test]
    with log.open('w') as output:
        result = subprocess.run(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},
                                stdout=output,stderr=subprocess.STDOUT,timeout=30)
    text = log.read_text()
    killed = result.returncode == 1 and ' failed' in text and 'ERROR collecting' not in text
    results.append(dict(name=name,exit_code=result.returncode,killed=killed,
                        test=test,log=str(log.relative_to(ROOT)),command=command))
    print(name, 'KILLED' if killed else 'RED', flush=True)
assert hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
(OUT/'mutations.json').write_text(json.dumps(dict(results=results,
    production_sources_unchanged=True,source_sha256=hashes),indent=2)+'\n')
assert all(r['killed'] for r in results)
