"""Prior art: C7 in-process AST mutation receipts (GRM/Python contributors,
2026). Reuse registered fault injection and pytest; new FIX5 prompt/budget
mutants. No prior art known to me for the exact pins. Never edits live core.
"""
import ast
from pathlib import Path
import sys

import pytest
from core import graft_arena as ga

mutant = sys.argv[1]
name = '_consolidation_prompts' if mutant == 'remove_enumeration' else 'consolidate'
source = Path(ga.__file__).read_text()
cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name=='ArenaCache')
fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name==name)
text = ast.unparse(fn)
if mutant == 'remove_enumeration':
    assert 'if need:' in text
    text = text.replace('if need:', 'if False:', 1)
    pin = 'test_r2_enumerated_facts_fold_cpu'
elif mutant == 'remove_scaling':
    assert '24 * len(need)' in text
    text = text.replace('24 * len(need)', '0', 1)
    pin = 'test_r2_default_budget_exhaustion'
else:
    raise ValueError('unregistered mutant')
ns = dict(vars(ga))
exec(compile(text, '<registered FIX5 mutant>', 'exec'), ns)
setattr(ga.ArenaCache, name, ns[name])
raise SystemExit(pytest.main(['-q', 'tests/test_grm_scout_fix5.py::'+pin]))
