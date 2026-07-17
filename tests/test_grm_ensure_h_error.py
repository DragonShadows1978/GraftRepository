import pytest

from core.graft_arena import ArenaCache, GraftPayloadMissingError


def _arena(grafts, node_loader=None):
    arena = ArenaCache.__new__(ArenaCache)
    arena.grafts = grafts
    arena.node_loader = node_loader
    return arena


def test_ensure_h_raises_named_error_for_unbackable_node():
    calls = []
    arena = _arena([{"h": None}], lambda i: calls.append(i))

    with pytest.raises(GraftPayloadMissingError) as raised:
        arena._ensure_h([0])

    assert raised.value.node_ids == (0,)
    assert raised.value.missing_node_ids == (0,)
    assert "0" in str(raised.value)
    assert calls == [0]
    assert arena.grafts[0]["h"] is None


def test_ensure_h_heals_host_and_disk_backed_nodes():
    host_payload = {"source": "host"}
    disk_payload = {"source": "disk"}
    sources = []
    arena = _arena([
        {"h": None, "host_payload": host_payload},
        {"h": None},
    ])

    def load(i):
        if arena.grafts[i].get("host_payload") is not None:
            sources.append((i, "host"))
            return arena.grafts[i]["host_payload"]
        sources.append((i, "disk"))
        return disk_payload

    arena.node_loader = load
    arena._ensure_h([0, 1])

    assert arena.grafts[0]["h"] is host_payload
    assert arena.grafts[1]["h"] is disk_payload
    assert sources == [(0, "host"), (1, "disk")]
    assert arena.page_ins == 2


def test_ensure_h_reports_only_unbackable_nodes_in_a_mixed_set():
    healed_payload = {"source": "host"}
    calls = []
    arena = _arena([
        {"h": None},
        {"h": None},
        {"h": {"source": "resident"}},
        {"h": None},
    ])

    def load(i):
        calls.append(i)
        return healed_payload if i == 1 else None

    arena.node_loader = load
    with pytest.raises(GraftPayloadMissingError) as raised:
        arena._ensure_h([0, 1, 2, 3])

    assert raised.value.node_ids == (0, 3)
    assert arena.grafts[1]["h"] is healed_payload
    assert arena.grafts[2]["h"] == {"source": "resident"}
    assert calls == [0, 1, 3]
