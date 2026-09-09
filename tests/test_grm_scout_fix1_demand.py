"""C1 CPU baseline: real observer/attempt, scripted model and mass only.

Prior art: GRM-SC2 CPU _ResumeArena (project contributors, 2026), reused;
new coverage drives the production observer rather than copying its abort rule.
"""
import sys
from types import SimpleNamespace

import pytest

from core import grm_demand
from tests.test_grm_sc2_calibration_early_abort import _ResumeArena, CARRIED


@pytest.fixture
def cpu_hooks(monkeypatch):
    kernel = lambda *args, **kwargs: None
    fake = SimpleNamespace(sink_attention_tc=kernel, sliding_sink_attention_tc=kernel)
    monkeypatch.setitem(sys.modules, 'core.gpt_oss20b_tc', fake)
    import core
    monkeypatch.setattr(core, 'gpt_oss20b_tc', fake, raising=False)
    clears = []
    monkeypatch.setattr('core.graft_arena.kv_graft.clear_injection',
                        lambda model: clears.append(model))
    return fake, kernel, clears


def run(masses, early_abort, stops=(), ngen=3):
    arena = _ResumeArena()
    obs = grm_demand.DemandObserver(arena, ngen, CARRIED, early_abort=early_abort)
    queue = iter(masses)
    def summarize(rows):
        mass = next(queue)
        return dict(mounted_mass=mass, physical_sink_mass=1-mass,
                    live_mass=0., learned_sink_mass=0.)
    obs._summarize_mass = summarize
    with obs:
        result = arena._attempt('q', [], ngen, False, list(stops))
    return arena, obs, result


def test_final_flush_only_fire_keeps_completed_answer(cpu_hooks):
    off_arena, off, expected = run([.9, .9, .9, .1], False)
    arena, obs, result = run([.9, .9, .9, .1], True)
    assert result == expected
    assert obs.finish() == off.finish()  # Before fix: captured 4 rows for ngen=3.
    assert obs.aborted_at is None
    assert len(obs.records) == 4 and obs.records[-1]['mounted_mass'] == .1
    assert obs.decision()['demand_fired'] is False
    assert arena.committed == off_arena.committed == 4
    assert arena.live_segs == off_arena.live_segs
    fake, kernel, clears = cpu_hooks
    assert fake.sink_attention_tc is kernel
    assert fake.sliding_sink_attention_tc is kernel
    assert len(clears) == 2


@pytest.mark.parametrize('stops', [(), ('3',), ('6',)])
def test_first_token_abort_resume_exact(cpu_hooks, stops):
    expected_arena, _, expected = run([.1, .9, .9, .9], False, stops)
    arena, obs, handle = run([.1, .9, .9, .9], True, stops)
    assert handle['abort_token_index'] == 0
    assert arena.live_segs == []
    rows = obs.finish()
    assert obs.decision()['demand_fired'] is True
    handle['out'] = [row['prediction_token_id'] for row in rows]
    result = arena._resume_attempt(handle, False, list(stops), False)
    assert result == expected
    assert arena.committed == expected_arena.committed
    assert arena.live_segs == expected_arena.live_segs
    with pytest.raises(RuntimeError, match='already resumed'):
        arena._resume_attempt(handle, False, list(stops), False)


@pytest.mark.parametrize('stop', ['3', '6', '9'])
@pytest.mark.parametrize('early_abort', [False, True])
def test_stop_token_has_no_flush(cpu_hooks, stop, early_abort):
    arena, obs, (answer, _) = run([.9, .9, .9, .1], early_abort, (stop,))
    assert arena.committed == int(stop)//3
    assert len(obs.finish()) == arena.committed
    assert obs.aborted_at is None
    assert answer == ' '.join(str(n) for n in range(3, int(stop), 3))


def test_single_token_flush_boundary(cpu_hooks):
    arena, obs, result = run([.9, .1], True, ngen=1)
    assert result[0] == '3'
    assert len(obs.finish()) == 1
    assert obs.aborted_at is None
    assert arena.committed == 2


def test_last_answer_position_still_aborts_and_resumes(cpu_hooks):
    expected_arena, _, expected = run([.9, .9, .1, .9], False)
    arena, obs, handle = run([.9, .9, .1, .9], True)
    assert handle['abort_token_index'] == 2
    assert len(obs.finish()) == 3
    assert obs.decision()['demand_fired'] is True
    handle['out'] = [row['prediction_token_id'] for row in obs.finish()]
    result = arena._resume_attempt(handle, False, [], False)
    assert result == expected
    assert arena.committed == expected_arena.committed == 4
    assert arena.live_segs == expected_arena.live_segs
