"""Prior art: GRM C7 CPU serving-path replay (GRM contributors, 2026).
Reuse real routing/attempts with numerical doubles. Exact source35 and frozen
whole-receipt controls are this order's pins; no prior art known to me for
this exact composition. CPU author evidence, not model answer quality.
"""
import json
import os
from pathlib import Path

import pytest

from scripts.grm_c7_diagnose import repository
from scripts.grm_e2e_session import _probe_ladder_chat, harmony_turn

BEFORE = Path(__file__).resolve().parents[1]/'artifacts/grm_scout_fix4/nonrecency_before.json'


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) made `margin_first` the shipped admission rule and
# turned F1/F2/F5/A1 on.  THIS SUITE'S ASSERTIONS ARE UNCHANGED: FIX-4's
# whole-receipt control (`nonrecency_before.json`) is a FROZEN RECEIPT of
# the pre-D2 world, recorded byte for byte, and re-blessing it would
# destroy the very control it exists to be.  `GRM_LEGACY_DEFAULTS=1`
# restores that world in one variable, which is exactly the case the
# umbrella was added for -- the order's own byte-identity gate.
@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    from core import grm_legacy_defaults as legacy_defaults
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")


def serve(repo, path, question):
    if path == 'core':
        return repo.arena.step(question, ngen=32, deposit=False, defer_memory=True)
    return _probe_ladder_chat(repo, question, topk=3, ngen=32, max_trips=1, defer_memory=True)


@pytest.mark.parametrize('path', ['core', 'ladder'])
@pytest.mark.parametrize('frame', ['recency', 'live'])
def test_c5_t33_source35_live_excluded_is_served(tmp_path, monkeypatch, path, frame):
    repo = repository(tmp_path/'repo', monkeypatch)
    a = repo.arena
    try:
        for i in range(35):
            a.feed(harmony_turn(f'Ordinary archive item {i}.', 'Recorded.'))
        if frame == 'live':
            a.ephemeral = False
            a.cache_deposits = False
        a.feed(harmony_turn('The current C5-Source-35 value is Basalt-811.', 'Recorded.'))
        a.recency_mounts = 1
        if frame == 'live':
            # Persistent live segment: retain the actual prefilled cache, no
            # nominal recency mount and no injection of source35 as a graft.
            a.ephemeral = False
            a.m.calls.clear()
            # Model double has no numerical attention; expose recorded live
            # payload at that numerical seam, and assert it is not remounted.
            a.m.injected = a.grafts[35]['text']
            a.m.remaining = a.encode('Basalt-811<|end|>')
            assert 35 in {g for g, _ in a.live_segs}
        answer, info = serve(repo, path, 'What is the current C5-Source-35 value?')
        assert not info.get('abstained', False)
        assert answer == 'Basalt-811'
        assert info['served_from'] == 'recency_mount'
        assert info['served_from_node_ids'] == [35]
        assert info['admission_identified_candidates'] == []
        assert 35 not in info['admission_rank_plan']
        assert a.cur_mounts == ([35] if frame == 'recency' else [])
    finally:
        repo.close()


@pytest.mark.parametrize('path', ['core', 'ladder'])
def test_identifier_binds_nothing_still_abstains(tmp_path, monkeypatch, path):
    repo = repository(tmp_path/'repo', monkeypatch)
    try:
        repo.arena.feed(harmony_turn('An ordinary archive record.', 'Recorded.'))
        answer, info = serve(repo, path, 'What is the current Missing-937 value?')
        assert answer == 'Not in memory: no stored record matches missing-937.'
        assert info['abstain_reason'] == 'identifier_unbound'
        assert info['abstained'] is True
        assert 'served_from' not in info
        assert not repo.arena.m.calls
    finally:
        repo.close()


def test_nonrecency_existing_fixture_receipt_byte_identical(tmp_path, monkeypatch):
    from scripts.grm_c7_common import FIX, read
    from scripts.grm_c7_run import effective_question
    f = read(FIX)
    result = {}
    for path in ('core', 'ladder'):
        repo = repository(tmp_path/path, monkeypatch)
        try:
            for t in f['turns'][:6]:
                repo.arena.feed(harmony_turn(t['user'], t['assistant']))
            p = next(p for p in f['probes'] if p['id'] == 'c7_fresh_0_d005')
            answer, info = serve(repo, path, effective_question(p['question']))
            assert answer == 'Basalt-811'
            assert repo.arena.cur_mounts == [5]
            assert info['admission_rank_plan'] == [5]
            # Prior art: FIX6 additive receipt contract (GRM contributors,
            # 2026). Assert the authorized new key independently, then compare
            # EVERY legacy receipt byte against the unchanged FIX4 baseline.
            assert info.pop('admission_rule') == 'all_tokens_bind'
            if path == 'ladder':
                assert info['_route_observation']['admission_profile'].pop('admission_rule') == 'all_tokens_bind'
            result[path] = {'answer': answer, 'info': info}
        finally:
            repo.close()
    data = (json.dumps(result, sort_keys=True, indent=2, default=str)+'\n').encode()
    if os.environ.get('GRM_FIX4_FREEZE_BEFORE') == '1':
        with BEFORE.open('xb') as stream:
            stream.write(data)
    assert data == BEFORE.read_bytes()
