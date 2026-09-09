"""Prior art: GRM C7 numerical doubles and source receipt replay (2026).
Reuse real consolidate/QC/coverage and numerical seams. New FIX5 stimuli
read only the prompt list; no prior art known to me for this exact suite.
CPU author evidence only, not a model-quality or relation-fidelity proof.
"""
import json
import os
from pathlib import Path
import re

import pytest

from scripts.grm_c7_diagnose import Model, repository
from scripts.grm_e2e_session import harmony_turn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_scout_fix5'
REG = json.loads((OUT/'registration.json').read_text())
CASES = REG['fold_starts']


class EnumeratedModel(Model):
    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            prompt = self.codec.decode(ids[0])
            lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
            # Independent stimulus extracts identifiers from enumerated source
            # spans only, never mounted text, expected answers or need_count.
            tokens = sorted(set(re.findall(r'\b[A-Za-z][\w-]*\d[\w-]*\b', ' '.join(lines))))
            self.fold_output = ('retained identifiers in the archival record include '
                                + ', '.join(tokens) + '.<|end|>')
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)


def seed(a, path):
    start = json.loads((ROOT/path).read_text())
    ids = []
    for source in start['sources']:
        idx = a.deposit(source['text'])
        # Observer null represented an absent kind (defaults to turn).
        if source['kind'] is not None:
            a.grafts[idx]['kind'] = source['kind']
        ids.append(idx)
    return start, ids


@pytest.mark.parametrize('path', CASES, ids=lambda p: Path(p).parent.name[:3])
def test_r2_enumerated_facts_fold_cpu(tmp_path, monkeypatch, path):
    repo = repository(tmp_path/'repo', monkeypatch)
    try:
        a = repo.arena
        a.m = EnumeratedModel(a.m.codec)
        start, ids = seed(a, path)
        result = a.consolidate(ids)
        need = a._fact_set([s['text'] for s in start['sources']])
        receipt = {'turn': start['turn'], 'result': a.last_consolidation_result,
                   'attempts': a.last_consolidation_attempts, 'digest_text': result[1],
                   'prompts': [c['input'] for c in a.m.calls if c['initial']],
                   'budget': max(120, 24*len(need))}
        if os.environ.get('GRM_FIX5_RECEIPTS'):
            target = Path(os.environ['GRM_FIX5_RECEIPTS'])
            target.mkdir(parents=True, exist_ok=True)
            with (target/f'{start["turn"]:03d}.json').open('x') as f:
                json.dump(receipt, f, indent=2, sort_keys=True)
        assert result[0] is not None, receipt
        assert a.last_consolidation_result['best_cov'] == 1.0
        assert a.last_consolidation_result['hit_count'] == len(need)
        assert a._digest_qc(result[1], None, forbid_lists=True)
        assert '\n' not in result[1]
        assert all(a.grafts[i]['retired'] for i in ids)
        for prompt in a._consolidation_prompts(False, [s['text'] for s in start['sources']]):
            assert f'Keep every one of these {len(need)} facts' in prompt
            lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
            assert need <= a._fact_set(lines)
            assert 'single prose archive block' in prompt
            assert 'Do not copy' in prompt
    finally:
        repo.close()


@pytest.mark.parametrize('path', CASES, ids=lambda p: Path(p).parent.name[:3])
def test_r2_default_budget_exhaustion(tmp_path, monkeypatch, path):
    repo = repository(tmp_path/'repo', monkeypatch)
    try:
        a = repo.arena
        start, ids = seed(a, path)
        a.m.fold_output = 'ordinary ' * 1000  # deliberate no-stop exhaustion
        a.consolidate(ids)
        n = len(a._fact_set([s['text'] for s in start['sources']]))
        assert len(a.m.calls) == 3 * max(120, 24*n)
        assert len(a.last_consolidation_attempts) == 3
    finally:
        repo.close()


@pytest.mark.parametrize('n', [0, 1, 5, 6, 10])
@pytest.mark.parametrize('explicit', [None, 19])
def test_default_budget_boundary_and_explicit_override(tmp_path, monkeypatch, n, explicit):
    repo = repository(tmp_path/'repo', monkeypatch)
    try:
        a = repo.arena
        idx = a.deposit('values ' + ' '.join(f'X-{i}' for i in range(n)))
        a.m.fold_output = 'ordinary ' * 1000
        a.consolidate([idx], ngen=explicit)
        assert len(a.m.calls) == 3 * (explicit if explicit is not None else max(120, 24*n))
        assert a.MIN_FOLD_KEEP == .70 and a.CONSOLIDATE_NGEN == 120
    finally:
        repo.close()


def test_zero_fact_fold_byte_identical(tmp_path, monkeypatch):
    records = {}
    for frame in ('harmony', 'legacy'):
        for scaffold in (False, True):
            repo = repository(tmp_path/f'{frame}-{scaffold}', monkeypatch)
            try:
                a = repo.arena
                a.prompt_template = harmony_turn if frame == 'harmony' else None
                a.TEXT_SCAFFOLD_CONSOLIDATION = scaffold
                idx = a.deposit('a quiet discussion about ordinary daily work.')
                assert a._fact_set([a.grafts[idx]['text']]) == set()
                a.m.fold_output = 'ordinary daily work continued calmly with useful discussion.<|end|>'
                result = a.consolidate([idx])
                records[f'{frame}-{scaffold}'] = {
                    'calls': a.m.calls, 'result': result, 'attempts': a.last_consolidation_attempts,
                    'summary': a.last_consolidation_result,
                    'nodes': [{k:g.get(k) for k in ('text','kind','retired','sources')} for g in a.grafts]}
            finally:
                repo.close()
    data = (json.dumps(records, indent=2, sort_keys=True)+'\n').encode()
    before = OUT/'zero_fact_before.json'
    if os.environ.get('GRM_FIX5_FREEZE_BEFORE') == '1':
        with before.open('xb') as f:
            f.write(data)
    assert data == before.read_bytes()
