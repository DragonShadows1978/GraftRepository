"""GRM-A1 — OFF byte-identity proof against the PARENT branch's own code.

The in-suite ``test_flag_off_byte_identical`` compares A1-OFF against
A1-explicitly-0 inside THIS tree.  That is necessary but not sufficient: both
arms run the same (modified) ``graft_repository.py``, so a change that
altered behaviour identically on both arms would pass.

This script closes that gap.  It runs the SAME scripted session twice —
once importing this worktree's ``core`` package with ``GRM_ALIAS_FOLD_MERGE``
unset, and once importing the PARENT worktree's ``core`` package (which has
no A1 code at all) — in two separate subprocesses, and compares the full
observable state byte for byte.

Observables captured: every node's text / kind / retired / ntok / no_fold /
sources / full metadata, every served answer, the complete model call log
(inputs, output tokens, injected text, position offsets), the fold history
and the consolidation receipts.  A changed FIX-5 prompt, a changed
generation-step count, a changed lineage field or a changed mount would all
move at least one of these.

Usage:
    python3 scripts/grm_a1_off_parity.py <parent_worktree> [--json OUT]

Prior art: the two-arm frozen-record comparison in
``tests/test_grm_scout_fix5.py::test_zero_fact_fold_byte_identical`` (GRM
contributors, 2026); taken: the "serialize every observable and compare
bytes" method.  New here: running the two arms against two DIFFERENT code
trees in separate processes, so the comparison witnesses the code change
itself rather than a flag within one tree.  No prior art known to me for this
exact composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SYSTEM = ('<|start|>system<|message|>You are ChatGPT. Reasoning: low. '
          'Valid channel: final.<|end|>')

#: The scripted session both arms run.  Deliberately alias-DENSE: if A1-OFF
#: were going to perturb anything, an alias edge beside its base is where.
SESSION = {
    'seed': [
        'The current C7-AliasBase-0 value is Jasper-711. '
        'The current C7-AliasBase-1 value is Jasper-712.',
        'C7-Signal-0 is an alias for C7-AliasBase-0.',
        'C7-Signal-1 is an alias for C7-AliasBase-1.',
        "Let's call Kestrel 'the Hauler' from now on.",
        "Kestrel's cargo allowance will be 37 crates.",
        'The current C7-Plain-0 value is Basalt-811.',
    ],
    'questions': [
        'What is the current value for C7-Signal-0? Reply only with the answer.',
        'What is the current value for C7-Signal-1? Reply only with the answer.',
        'What is the current C7-Plain-0 value? Reply only with the answer.',
        "What did we settle on for the Hauler's cargo allowance?",
    ],
    'fold_sources': [0, 5],
    'correction': ['current C7-Plain-0 value is Basalt-811',
                   'The current C7-Plain-0 value is Onyx-911.'],
}

ARM = r'''
import json, os, re, sys, tempfile
sys.path.insert(0, ROOT)
os.environ.pop("GRM_ALIAS_FOLD_MERGE", None)
from _pytest.monkeypatch import MonkeyPatch
from scripts.grm_c7_diagnose import Model, repository

SYSTEM = SYSTEM_TEXT

def harmony(user):
    return (SYSTEM + "<|start|>user<|message|>" + user + "<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Recorded.<|end|>")

class EnumeratedModel(Model):
    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            prompt = self.codec.decode(ids[0])
            lines = re.findall(r"^\[source \d+\] (.+)$", prompt, re.M)
            if lines:
                self.fold_output = ("The archived record states that "
                                    + " ".join(l.rstrip(".") + "." for l in lines)
                                    + "<|end|>")
            else:
                self.fold_output = None
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)

session = json.loads(SESSION_JSON)
mp = MonkeyPatch()
repo = repository(tempfile.mkdtemp() + "/repo", mp)
repo.arena.m = EnumeratedModel(repo.arena.m.codec)
a = repo.arena
for text in session["seed"]:
    i = a.deposit(harmony(text))
    a.grafts[i]["kind"] = "turn"
repo._sync_lifecycle()
a.m.fold_output = None
answers = [a.step(q, ngen=8, deposit=False)[0] for q in session["questions"]]
fold = a.consolidate(session["fold_sources"])
repo.correct_memory(*session["correction"])
answers += [a.step(q, ngen=8, deposit=False)[0] for q in session["questions"]]

out = {
    "answers": answers,
    "fold": list(fold),
    "consolidation": getattr(a, "last_consolidation_result", None),
    "attempts": getattr(a, "last_consolidation_attempts", None),
    "fold_history": repo.fold_history,
    "calls": a.m.calls,
    "nodes": [{k: g.get(k) for k in
               ("text", "kind", "retired", "ntok", "no_fold", "sources",
                "metadata", "node_id", "durable", "dirty")}
              for g in a.grafts],
}
sys.stdout.write(json.dumps(out, indent=1, sort_keys=True, default=str))
'''


def run_arm(root: Path) -> str:
    script = ('ROOT = %r\nSYSTEM_TEXT = %r\nSESSION_JSON = %r\n' % (
        str(root), SYSTEM, json.dumps(SESSION))) + ARM
    env = dict(os.environ)
    env.pop('GRM_ALIAS_FOLD_MERGE', None)
    env['PYTHONHASHSEED'] = '0'
    proc = subprocess.run([sys.executable, '-c', script], cwd=str(root),
                          capture_output=True, text=True, env=env, timeout=540)
    if proc.returncode != 0:
        raise SystemExit(f'arm {root} failed:\n{proc.stderr[-4000:]}')
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('parent', help='parent worktree (pre-A1 core)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    parent = Path(args.parent).resolve()

    mine = run_arm(here)
    theirs = run_arm(parent)
    same = mine == theirs
    receipt = {
        'schema': 'grm.a1.off_parity.v1',
        'arm_a1_off': {'root': str(here),
                       'sha256': hashlib.sha256(mine.encode()).hexdigest(),
                       'bytes': len(mine)},
        'arm_parent': {'root': str(parent),
                       'sha256': hashlib.sha256(theirs.encode()).hexdigest(),
                       'bytes': len(theirs)},
        'byte_identical': same,
        'session': SESSION,
    }
    if not same:
        import difflib
        receipt['first_difference'] = list(difflib.unified_diff(
            theirs.splitlines(), mine.splitlines(),
            'parent', 'a1_off', lineterm='', n=2))[:60]
    text = json.dumps(receipt, indent=1, sort_keys=True) + '\n'
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
