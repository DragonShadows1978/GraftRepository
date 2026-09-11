"""GRM-A1 — per-digest width receipts for every registered alias fixture row.

Tokenizes each merged digest against the arena width BEFORE any single-mount
claim, as the order requires, and records the PRODUCTION arithmetic the merge
exists to defeat (RD2: edge 49 + base 61 = 110 > width 96) alongside the CPU
measurement.

The two token scales are reported SEPARATELY and never mixed: the CPU codec
(``scripts.grm_c7_diagnose.Codec``) splits on whitespace and its lengths are
NOT GPT-OSS geometry.  The CPU column proves the merge produces ONE node the
arena can seat; the production column is the registered receipt arithmetic
from RD2.  Confirming the merged digest's GPT-OSS token count is the lead's
registered GPU contrast.

Prior art: FIX-5's receipt scripts (GRM contributors, 2026) for the
per-fixture-row receipt shape.  No prior art known to me for this exact
width-receipt composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import grm_alias_fold as af  # noqa: E402

FIXTURE_PATH = ROOT / 'artifacts/grm_a1/alias_fixture.json'


def _harness():
    from _pytest.monkeypatch import MonkeyPatch
    from scripts.grm_c7_diagnose import Model, repository

    SYSTEM = ('<|start|>system<|message|>You are ChatGPT. Reasoning: low. '
              'Valid channel: final.<|end|>')

    def harmony(user):
        return (SYSTEM + '<|start|>user<|message|>' + user + '<|end|>'
                + '<|start|>assistant<|channel|>final<|message|>'
                + 'Recorded.<|end|>')

    class EnumeratedModel(Model):
        def __call__(self, ids, kv_caches=None, **kwargs):
            if kv_caches is None:
                prompt = self.codec.decode(ids[0])
                lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
                if lines:
                    self.fold_output = (
                        'The archived record states that '
                        + ' '.join(l.rstrip('.') + '.' for l in lines)
                        + '<|end|>')
                else:
                    self.fold_output = None
            return super().__call__(ids, kv_caches=kv_caches, **kwargs)

    def build(texts):
        mp = MonkeyPatch()
        mp.setenv('GRM_ALIAS_FOLD_MERGE', '1')
        repo = repository(Path(tempfile.mkdtemp()) / 'repo', mp)
        repo.arena.m = EnumeratedModel(repo.arena.m.codec)
        for text in texts:
            idx = repo.arena.deposit(text)
            repo.arena.grafts[idx]['kind'] = 'turn'
            repo.arena.grafts[idx]['no_fold'] = True
        repo._sync_lifecycle()
        return repo

    return build, harmony


def group_rows(fixture):
    build, harmony = _harness()
    out = []

    # --- RD2: the exact recorded node texts --------------------------------
    repo = build([fixture['rd2']['base_text']]
                 + [r['edge_text'] for r in fixture['rd2']['rows'][:2]])
    rd2_merges = repo.alias_fold_pass()
    for merge in rd2_merges:
        out.append({'group': 'rd2', 'alias': merge.get('alias'),
                    'reason': merge['reason'], 'lineage': merge.get('lineage'),
                    'width': merge.get('width'),
                    'base_shared_with_edges': merge.get(
                        'base_shared_with_edges'),
                    'digest_text': merge.get('digest_text')})
    repo.close()

    # --- C7 r3: the fixture's own oracle source texts ----------------------
    texts, seen = [], set()
    for row in fixture['c7_r3']['rows']:
        for src in row['oracle_source_texts']:
            if src not in seen:
                seen.add(src)
                texts.append(harmony(src))
    repo = build(texts)
    for merge in repo.alias_fold_pass():
        out.append({'group': 'c7_r3', 'alias': merge.get('alias'),
                    'reason': merge['reason'], 'lineage': merge.get('lineage'),
                    'width': merge.get('width'),
                    'base_shared_with_edges': merge.get(
                        'base_shared_with_edges'),
                    'digest_text': merge.get('digest_text')})
    repo.close()

    # --- LT1: natural-language alias declarations --------------------------
    texts, seen = [], set()
    for row in fixture['lt1']['rows']:
        for src in row['source_texts']:
            if src not in seen:
                seen.add(src)
                texts.append(harmony(src))
    repo = build(texts)
    for merge in repo.alias_fold_pass():
        out.append({'group': 'lt1', 'alias': merge.get('alias'),
                    'reason': merge['reason'], 'lineage': merge.get('lineage'),
                    'width': merge.get('width'),
                    'base_shared_with_edges': merge.get(
                        'base_shared_with_edges'),
                    'digest_text': merge.get('digest_text')})
    repo.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=str(
        ROOT / 'artifacts/grm_a1/width_receipts.json'))
    args = ap.parse_args()

    fixture = json.loads(FIXTURE_PATH.read_text())
    rows = group_rows(fixture)
    fits = [r for r in rows if (r['width'] or {}).get('fits')]
    receipt = {
        'schema': 'grm.a1.width_receipts.v1',
        'fixture_sha256': hashlib.sha256(
            FIXTURE_PATH.read_bytes()).hexdigest(),
        'token_scales': {
            'cpu_codec': ('scripts.grm_c7_diagnose.Codec — whitespace split. '
                          'NOT GPT-OSS geometry. Proves the merge yields ONE '
                          'seatable node; claims no production token count.'),
            'production': ('RD2 receipts, GPT-OSS geometry: alias edge 49 + '
                           'alias base 61 = 110 > arena width 96. This is the '
                           'arithmetic co-mount (FIX-7) could not defeat.'),
        },
        'production_widths': fixture['production_widths'],
        'digests': rows,
        'summary': {
            'digests': len(rows),
            'fits_under_width': len(fits),
            'over_width': len(rows) - len(fits),
            'all_single_mount_claimable': len(fits) == len(rows),
        },
    }
    text = json.dumps(receipt, indent=1, sort_keys=True,
                      ensure_ascii=False) + '\n'
    Path(args.json).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
