"""GRM-P1 FOLLOW-UP — the native-publication defect, reproduced on CPU.

The first GRM-P1 GPU smoke went RED at the first recall:

    scripts/grm_e2e_session.py:1889 _probe_ladder_chat
      -> :1628 _serve
      -> core/graft_arena.py:4447 _attempt
      -> :2377 _commit_native_mount
      -> :2369 _native_mount_ids
      RuntimeError: graft 0 has no native_node_id

CAUSE (surface, not core). ``native_node_id`` is assigned by
``GraftRepository._native_sync_node``, reached from ``_mark_dirty(payload=
True)`` inside ``repo._mark_mutations(before)``, which production reaches
through ``repo.runtime._finish_turn_event(...)`` — the funnel
``grm_e2e_session._probe_finish_deposit`` calls on every ordinary chat
turn.  The GRM-P1 surface deposited its EB1 complete turn with
``arena.feed()`` (correct, and what ``grm_lt1_worker.execute`` does) but
then stopped there, so nothing ever published the fed node to the native
store.  The next turn routed to it, tried to mount it, and the native
commit raised.

The LT1 battery met the same wall on 2026-09-09 and answered it with a
HARNESS workaround (``grm_lt1_amendment4.install_native_publication``
monkey-patches ``_commit_native_mount`` to publish lazily at mount time).
That is right for a fixture replayer and wrong for a product, so the fix
here is to call the production funnel.  ``core/`` is UNCHANGED.

These tests are the RED->GREEN receipt: ``test_feed_without_the_funnel_
reproduces_the_gpu_crash`` fails the way the GPU did when the funnel is
skipped, and the rest show the shipped ``ChatSession.ask`` does not.

Prior art: ``grm_lt1_amendment4.install_native_publication`` (GRM
contributors, 2026) — taken: the diagnosis (fed nodes lack native ids).
Not taken: the monkey-patch itself.  The stub store below follows the
``FakeNativeStore`` shape in ``tests/test_grm_fold_recovered_guard.py``
(GRM, 2026).  No prior art known to me for this exact fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_chat                                    # noqa: E402
from scripts.grm_profile import resolve_profile                 # noqa: E402


class StubNativeStore:
    """The narrowest store that exercises the publication contract.

    ``_native_mount_ids`` only requires ``commit_mount`` to exist for the
    commit path to run; ``_native_sync_node`` needs ``add_node`` to mint an
    id and ``set_route``/``set_metadata`` to accept it.  Nothing here
    simulates the real runtime — this is a boundary double whose only job
    is to make "was this graft published?" observable on CPU.
    """

    def __init__(self) -> None:
        self.nodes: dict[int, str] = {}
        self.commits: list[tuple[list[int], int]] = []
        self._next = 0

    def add_node(self, text, blob, ntok=0):
        node_id = self._next
        self._next += 1
        self.nodes[node_id] = text
        return node_id

    def set_route(self, node_id, key, lexical):
        pass

    def set_metadata(self, node_id, *args, **kwargs):
        pass

    def commit_mount(self, node_ids, mount_tokens):
        # The real store would fail on an unknown id; the arena raises
        # before reaching here if any graft lacked a native id.
        for node_id in node_ids:
            if int(node_id) not in self.nodes:
                raise RuntimeError(f"unknown native node {node_id}")
        self.commits.append(([int(i) for i in node_ids], int(mount_tokens)))

    # -- incidental surface ------------------------------------------------
    # Reached by the turn funnel (librarian, durability, paging) but not
    # part of the publication contract. Each returns the TYPE the caller
    # expects — a blanket ``None``-returning ``__getattr__`` makes the stub
    # lie (the librarian iterates ``foldable_nodes``' result) and turns a
    # fixture into a source of false failures.
    #
    # ``add_structured_node`` and the richer route setters are ABSENT on
    # purpose: ``_native_sync_node`` and ``_native_set_route`` pick their
    # branch with ``hasattr``, so defining them here would silently take a
    # branch this fixture does not intend to exercise.

    def mark_durable(self, node_id):
        pass

    def set_active(self, *args, **kwargs):
        pass

    def set_no_fold(self, *args, **kwargs):
        pass

    def set_provenance(self, *args, **kwargs):
        pass

    def set_graph_edges(self, *args, **kwargs):
        pass

    def clear_route(self, node_id):
        pass

    def clear_payload(self, node_id):
        pass

    def foldable_nodes(self, kind, excluded):
        return []

    def dirty_node_ids(self):
        return []

    def filter_active_nodes(self, node_ids):
        return list(node_ids)

    def stats(self):
        return {}

    def close(self):
        pass


@pytest.fixture()
def native_session(tmp_path):
    """A fake-model session wired to a stub native store."""
    resolved = resolve_profile(selection="eb1_c2")
    session = grm_chat.ChatSession(tmp_path / "session", resolved, fake=True)
    session.open()
    store = StubNativeStore()
    session.repo.native_store = store
    session.repo.arena.native_store = store
    yield session, store
    grm_chat._save_fake_codec(session)
    session.close()


def _mount(session, picks):
    """Drive the real ``_commit_native_mount`` — the GPU crash's call."""
    arena = session.repo.arena
    arena._commit_native_mount(
        picks, sum(int(arena.grafts[i].get("ntok", 0)) for i in picks))


def test_feed_without_the_funnel_reproduces_the_gpu_crash(native_session):
    """RED: feed() alone leaves a graft unpublished; the mount raises."""
    session, store = native_session
    from scripts.grm_e2e_session import harmony_turn

    arena = session.repo.arena
    idx = int(arena.feed(harmony_turn(
        "The current orion pin value is Auric-4-Alpha.", "Understood.")))
    arena.grafts[idx]["kind"] = "turn"
    # No runtime._finish_turn_event -> no _mark_mutations -> no native id.
    assert arena.grafts[idx].get("native_node_id") is None
    assert store.nodes == {}

    with pytest.raises(RuntimeError, match="has no native_node_id"):
        _mount(session, [idx])


def test_the_turn_funnel_publishes_the_fed_node(native_session):
    """GREEN: the production funnel assigns the id feed() did not."""
    session, store = native_session
    from scripts.grm_e2e_session import harmony_turn

    repo = session.repo
    arena = repo.arena
    before = repo._snapshot_state()
    idx = int(arena.feed(harmony_turn(
        "The current orion pin value is Auric-4-Alpha.", "Understood.")))
    arena.grafts[idx]["kind"] = "turn"
    repo.runtime._finish_turn_event("chat", before, autosave=True)

    assert arena.grafts[idx].get("native_node_id") is not None
    assert store.nodes, "the fed node reached the native store"
    _mount(session, [idx])                          # no raise
    assert store.commits[-1][0] == [int(arena.grafts[idx]["native_node_id"])]


def test_the_surface_publishes_fed_nodes(native_session):
    """The shipped ``ChatSession.ask`` publishes, with no harness patch."""
    session, store = native_session
    row = session.ask("The current orion pin value is Auric-4-Alpha.")
    node_id = row["deposited_node_id"]
    assert node_id is not None
    graft = session.repo.arena.grafts[node_id]
    assert graft.get("native_node_id") is not None
    assert int(graft["native_node_id"]) in store.nodes


def test_every_deposited_turn_is_mountable(native_session):
    """The real regression: turn 4 must be able to mount turns 1-3.

    This is the GPU smoke's failing shape (three deposits, then a recall
    that routes to them), run on CPU against the real
    ``_commit_native_mount``.
    """
    session, store = native_session
    for text in ("The current orion pin value is Auric-4-Alpha.",
                 "The current lyra dock value is Nadir-1-Delta.",
                 "The current cypher bridge value is Vortex-3-Sierra."):
        session.ask(text)

    arena = session.repo.arena
    deposited = [i for i, g in enumerate(arena.grafts) if not g.get("retired")]
    assert deposited, "nothing was deposited"
    for i in deposited:
        assert arena.grafts[i].get("native_node_id") is not None, (
            f"graft {i} would raise at mount time")
    _mount(session, deposited)

    row = session.ask("What is the current orion pin value?")
    assert row["mounted_ids"], "the recall turn mounted nothing"


def test_surface_refuses_a_silently_unpublished_node(native_session):
    """The surface's own guard fires if the funnel ever stops publishing."""
    session, _store = native_session
    original = session.repo.runtime._finish_turn_event

    def funnel_that_does_not_publish(*args, **kwargs):
        return None                      # simulate a regressed funnel

    session.repo.runtime._finish_turn_event = funnel_that_does_not_publish
    try:
        with pytest.raises(RuntimeError, match="NATIVE_PUBLICATION_FAILED"):
            session.ask("The current orion pin value is Auric-4-Alpha.")
    finally:
        session.repo.runtime._finish_turn_event = original


def test_no_native_store_is_not_an_error(tmp_path):
    """The CPU fake has no native store at all; the guard must not fire."""
    resolved = resolve_profile(selection="eb1_c2")
    session = grm_chat.ChatSession(tmp_path / "session", resolved, fake=True)
    session.open()
    try:
        assert getattr(session.repo, "native_store", None) is None
        row = session.ask("The current orion pin value is Auric-4-Alpha.")
        assert row["deposited_node_id"] is not None
    finally:
        grm_chat._save_fake_codec(session)
        session.close()
