"""Prior art: C7/A7 CPU controller, hash and reader isolation tests (2026).
New FIX8 source-origin and admission replay boundaries; no prior art known
for this exact composition. Author checks, not blind verification or quality.
"""
from contextlib import nullcontext
from pathlib import Path
import copy
import pytest
from scripts import grm_scout_fix8_replay as run


def test_registration_exact_cohort_and_original_middle():
    r=run.verify();old=run.verify_middle(payloads=False)
    qs=run.read(run.OUT/'requests.json')
    assert len(qs)==24 and sum(q['probe']['expected']=='UNKNOWN' for q in qs)==10
    assert sum(q['probe']['class']=='fresh' and q['probe']['expected']!='UNKNOWN' for q in qs)==8
    assert sum(q['probe']['class']=='folded' and q['probe']['expected']!='UNKNOWN' for q in qs)==6
    assert r['middle_registration_sha256']==run.sha(run.A7/'registration.json')
    assert len(old['batches'])==6 and old['maximum_reserved_seconds']==1680
    assert r['lease_seconds']==280 and r['gpu_cap_seconds']==1800
    assert run.summary()['status']=='NOT_MEASURED'


def test_input_drift_rejected(monkeypatch):
    original=run.sha
    monkeypatch.setattr(run,'sha',lambda p:'bad' if Path(p)==run.ROOT/'core/grm_admission.py' else original(p))
    with pytest.raises(ValueError,match='FIX8_INPUT_SHA_MISMATCH'):run.verify()


def test_middle_archive_drift_rejected(monkeypatch):
    original=run.sha
    monkeypatch.setattr(run,'sha',lambda p:'bad' if Path(p)==run.OUT/'before/core/grm_admission.py' else original(p))
    with pytest.raises(ValueError,match='A7_ARCHIVED_INPUT_SHA_MISMATCH'):run.verify_middle(payloads=False)


def test_reader_routes_and_does_not_receive_expected(tmp_path,monkeypatch):
    from scripts.grm_c7_diagnose import repository
    repo=repository(tmp_path/'repo',monkeypatch)
    try:
        repo.arena.feed('The current C7‑Fresh‑0 value is Basalt‑811.')
        q=copy.deepcopy(run.read(run.OUT/'requests.json')[0])
        q['probe']['expected']='SCORER_ONLY_SECRET_583'
        value=run.serve(repo,q,tmp_path/'result')
        assert value['admitted'] and value['mounted_ids']==[0]
        assert not value['identifier_unbound']
        assert repo.arena.m.calls
        assert all('SCORER_ONLY_SECRET_583' not in c['input']+c['injected'] for c in repo.arena.m.calls)
        assert (tmp_path/'result'/(q['probe_id']+'.json')).exists()
        assert repo.arena.cur_mounts==[] and len(repo.arena.grafts)==1
    finally:repo.close()


def test_owner_and_orphan_fail_closed(tmp_path,monkeypatch):
    r=run.verify();monkeypatch.setattr(run,'verify',lambda:r);monkeypatch.setattr(run,'OUT',tmp_path)
    (tmp_path/'replay.active').write_text('another owner')
    with pytest.raises(FileExistsError):run.batch('F1')
    assert (tmp_path/'replay.active').read_text()=='another owner'
    (tmp_path/'replay.active').unlink()  # this test created it
    run.write(tmp_path/'gpu/F1/reservation.json',{'seconds':280})
    with pytest.raises(ValueError,match='ORPHAN_RESERVATION_STOP'):run.batch('F1')
    assert not (tmp_path/'replay.active').exists()


def test_failed_lease_charged_and_never_retried(tmp_path,monkeypatch):
    r=run.verify();monkeypatch.setattr(run,'verify',lambda:r);monkeypatch.setattr(run,'OUT',tmp_path)
    from scripts import grm_cmc1_gpu_arms as leases
    def busy(*_):raise RuntimeError('BUSY_TEST_NO_GPU')
    monkeypatch.setattr(leases,'gpu_lease',busy)
    monkeypatch.setattr(run.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='BUSY_TEST_NO_GPU'):run.batch('F1')
    c=run.read(tmp_path/'gpu/F1/controller.json')
    assert c['status']=='FAILED' and c['charged_seconds']==280
    assert not (tmp_path/'replay.active').exists()
    with pytest.raises(ValueError,match='FAILED_CAMPAIGN_STOP'):run.batch('F1')


def test_registered_no_live_core_replacement():
    source=Path(run.__file__).read_text()
    assert 'core.__path__.insert' in source
    assert 'MIDDLE_CORE_ORIGIN_MISMATCH' in source
    assert 'MIDDLE_REQUIRES_FRESH_PROCESS' in source
