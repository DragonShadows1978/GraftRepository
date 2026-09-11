"""Run unchanged tests with selected modules loaded from a source snapshot.
Prior art: Python importlib source loaders (Python project documentation, 2026;
https://docs.python.org/3/library/importlib.html); ordinary source
substitution for regression/mutation checks. No new algorithm claimed.
"""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
overlay = Path(sys.argv[1]).resolve()

class SnapshotLoader(importlib.abc.Loader):
    def __init__(self, path, original):
        self.path, self.original = path, original

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.original)
        exec(compile(self.path.read_bytes(), str(self.original), 'exec'), module.__dict__)

class SnapshotFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        relative = Path(*fullname.split('.')).with_suffix('.py')
        source = overlay/relative
        if source.is_file():
            return importlib.util.spec_from_loader(fullname, SnapshotLoader(source, ROOT/relative))

sys.meta_path.insert(0, SnapshotFinder())
import pytest
raise SystemExit(pytest.main(sys.argv[2:]))
