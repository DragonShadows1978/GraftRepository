"""Prior art: GRM C7 CPU doubles and LT1 checkpoint contracts (2026), reused.
No prior art known to me for this exact failure/adversarial composition.
"""
from pathlib import Path
import pytest
from scripts import grm_lt1 as lt
from scripts import grm_lt1_worker as worker
from scripts.grm_lt1_amendment4_replay import restored, feed_cell_prefix


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    repo = restored(tmp_path, monkeypatch)
    fed = feed_cell_prefix(repo)
    # Before implementation this deliberately exercises the original failure.
    if hasattr(worker, 'install_native_publication'):
        worker.install_native_publication(repo, fed, tmp_path, {'turn':39,'phase':'session'})
    try:
        yield repo, fed, tmp_path
    finally:
        repo.close()


def test_fed_mount_publishes_and_commits_original_failure(loaded):
    repo, fed, out = loaded
    assert repo.arena.grafts[25].get('native_node_id') is None
    repo.arena._commit_native_mount([25], repo.arena.grafts[25]['ntok'])
    assert repo.arena.grafts[25]['native_node_id'] == 25
    assert repo.native_store.stats().nodes == 26
    assert lt.read(out/'native_publication.jsonl')['graft_id'] == 25


def test_old_nodes_and_unmounted_fed_nodes_do_not_publish(loaded):
    repo, fed, out = loaded
    repo.arena._commit_native_mount([11], repo.arena.grafts[11]['ntok'])
    repo.arena._commit_native_mount([], 0)
    assert repo.native_store.stats().nodes == 25
    assert all(g.get('native_node_id') is None for g in fed.values())
    assert not (out/'native_publication.jsonl').exists()


@pytest.mark.parametrize('case', ['checkpoint', 'untracked', 'replaced', 'mapped', 'payload', 'split'])
def test_refuses_unknown_missing_identity(loaded, case):
    repo, fed, out = loaded
    i = 25
    if case == 'checkpoint':
        i = 0
        repo.arena.grafts[0].pop('native_node_id')
    elif case == 'untracked':
        fed.pop(i)
    elif case == 'replaced':
        repo.arena.grafts[i] = dict(repo.arena.grafts[i])
    elif case == 'mapped':
        repo._native_node_ids[i] = 25
    elif case == 'payload':
        fed[i]['h'] = None
    elif case == 'split':
        fed[i]['metadata'] = {'width_guard_child':True}
    with pytest.raises(RuntimeError, match='LT1_UNPUBLISHED_NODE_OUTSIDE_FEED'):
        repo.arena._commit_native_mount([i], 1)
    assert repo.native_store.stats().nodes == 25


def test_repeated_mount_does_not_duplicate_registration(loaded):
    repo, fed, out = loaded
    for _ in range(2):
        repo.arena._commit_native_mount([25], repo.arena.grafts[25]['ntok'])
    assert repo.native_store.stats().nodes == 26
    assert len((out/'native_publication.jsonl').read_text().splitlines()) == 1


def test_native_unavailable_keeps_original_cpu_path(loaded):
    repo, fed, out = loaded
    store = repo.arena.native_store
    repo.arena.native_store = None
    repo.arena._commit_native_mount([25], 1)
    assert fed[25].get('native_node_id') is None
    repo.arena.native_store = store


def test_replay_retains_checkpoint_ids_and_names_graft25(loaded):
    repo, fed, out = loaded
    assert [g['native_node_id'] for g in repo.arena.grafts[:25]] == list(range(25))
    assert repo._native_checkpoint_loaded
    assert '89 ingots' in fed[25]['text'] and 'Foundry' in fed[25]['text']
    assert all(not g.get('sources') for g in fed.values())


def test_resume_starts_at_failed_cell_and_keeps_charge():
    from scripts import grm_lt1_amendment4 as a4
    r = lt.verify()
    assert worker.pending(r)['id'] == 'A-033-040'
    assert a4.cell_directory(lt.RUN/'cells','A-033-040').name == a4.RETRY
    controllers = list((lt.RUN/'cells').glob('*/controller.json'))
    expected = sum(lt.read(p)['charged_seconds'] for p in controllers)
    assert worker.accounting() == pytest.approx(expected)
    assert a4.cell_directory(lt.RUN/'cells','A-025-032').name == 'A-025-032'


def test_partial_failed_probes_are_not_scored():
    result = lt.summary('A')
    assert result['measured_recalls'] == 7
    assert not result['complete']


@pytest.mark.parametrize('field,value,error', [
    ('parent_sha256','0'*64,'CHAIN'), ('resume_at','A-041-048','PROTOCOL'),
    ('budget_seconds',20000,'PROTOCOL'), ('retry_directory','arbitrary','PROTOCOL'),
    ('retain_completed',[],'PROTOCOL'), ('new_inputs',{},'SCOPE')])
def test_rehashed_resume_forgery_refused(tmp_path, monkeypatch, field, value, error):
    import json
    from scripts import grm_lt1_amendment4 as a4
    r = lt.read(a4.REG); r[field] = value
    p = tmp_path/'resume.json'; p.write_text(json.dumps(r))
    p.with_suffix('.sha256').write_text(lt.sha(p))
    monkeypatch.setattr(a4,'REG',p)
    with pytest.raises(ValueError, match='AMENDMENT4_.*'+error):
        lt.verify()


def test_stale_checksum_refused(tmp_path, monkeypatch):
    from scripts import grm_lt1_amendment4 as a4
    p = tmp_path/'resume.json'; p.write_bytes(a4.REG.read_bytes()+b' ')
    p.with_suffix('.sha256').write_text(lt.sha(a4.REG))
    monkeypatch.setattr(a4,'REG',p)
    with pytest.raises(ValueError, match='AMENDMENT4_SHA_MISMATCH'):
        lt.verify()


def test_unknown_red_and_orphan_still_block(tmp_path):
    import json
    d = tmp_path/'cells/other'; d.mkdir(parents=True)
    (d/'reservation.json').write_text('{}')
    with pytest.raises(ValueError, match='ORPHAN_RESERVATION'):
        worker.accounting(tmp_path)
    (d/'controller.json').write_text(json.dumps({'status':'RED','charged_seconds':3}))
    with pytest.raises(ValueError, match='PRIOR_CELL_RED'):
        worker.accounting(tmp_path)


def test_actual_attempt_clears_original_failure_without_changing_pick(loaded):
    repo, fed, out = loaded
    question = "What did we settle on for Breakwater's map position?"
    answer, info = repo.arena._attempt(question, [25], 1, False, (), defer_memory=True)
    assert repo.arena.cur_mounts == [25]
    assert repo.arena.grafts[25]['native_node_id'] == 25
    assert '89 ingots' in repo.arena.grafts[25]['text']
    assert repo.native_store.stats().nodes == 26


def test_retry_refuses_second_red_and_never_moves_original(tmp_path, monkeypatch):
    from scripts import grm_lt1_amendment4 as a4
    monkeypatch.setattr(a4,'ROOT',tmp_path)
    cells=tmp_path/'artifacts/grm_lt1/amendment2/run_margin_first/cells'
    retry=a4.cell_directory(cells,a4.FAILED)
    assert retry.name == a4.RETRY
    assert a4.accepted_failed_charge(retry,{'status':'RED'}) is None
    assert a4.cell_directory(cells,'A-041-048').name == 'A-041-048'
