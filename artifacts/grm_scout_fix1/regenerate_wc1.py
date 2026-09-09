"""CPU-only replay of existing WC1 assembler; no experiment or scoring changes.
Prior art: WC1 results assembler (project contributors, 2026), reused verbatim.
Only artifact input/output locations are rebound for this read-only replay.
"""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import grm_wc1_sweep_gpu as sweep

stage = sys.argv[1]
assert stage in ('before', 'after')
out = ROOT/'artifacts/grm_scout_fix1'
source = (out/'before/scripts/grm_wc1_results.py' if stage == 'before'
          else ROOT/'scripts/grm_wc1_results.py')
spec = importlib.util.spec_from_file_location('wc1_replay', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = Path('/mnt/ForgeRealm/GraftRepository/artifacts/grm_wc1_opus')
sweep.RUN_ROOT = inputs/'runs'
sweep.REGISTRATION = inputs/'registration.json'
module.ROOT = ROOT
module.ARTIFACT_DIR = out
module.RESULTS = out/f'wc1_{stage}.json'
with (out/f'wc1_{stage}.txt').open('w') as log, contextlib.redirect_stdout(log):
    module.main(['--print-table'])
# Pin every read-only source receipt, including logs not listed by old assembler.
paths = sorted(p for p in inputs.rglob('*') if p.is_file() and
               (p.name in ('instrumentation.jsonl', 'manifest.json', 'registration.json')
                or (p.parent.name.startswith('w') and p.suffix == '.json')))
records = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out/f'wc1_{stage}_input_sha256.json').write_text(json.dumps(records, indent=2)+'\n')
print((out/f'wc1_{stage}.txt').read_text())
