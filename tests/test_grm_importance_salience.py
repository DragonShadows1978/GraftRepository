"""GRM-IMPORTANCE WO-3 (S2 PROSPECT): silent self-report salience pass.

Two tiers (docs/GRM_IMPORTANCE_PLAN.md is law):

  1. CPU UNIT TESTS (default, pytest, no GPU, no model load): rubric-score
     parsing, manifest write of ONLY the s2_salience key, opt-in flag off
     -> the pass never gets scheduled or run. The generation call itself
     is mocked (monkeypatched _s2_generate) — no tensor_cuda import here.

  2. GPU SMOKE — script-style, gated behind --run-gpu, NOT executed by
     pytest collection and NOT executed by this authoring agent per work
     order (GPU discipline: agents deliver code + CPU-verifiable tests,
     the lead runs GPU gates serially, one model resident at a time). The
     lead runs:

       python3 tests/test_grm_importance_salience.py --run-gpu

     One short scripted conversation, idle() fires with s2_salience_enabled
     =True, asserts a real 0-3 score lands in the manifest, and that arena
     state (node count/kind/retired/live cache shape) is identical before
     and after the scoring pass ran.
"""
import argparse
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graft_repository import (
    GraftRepository, S2_SALIENCE_PROMPT,
)


# ============================================================================
# CPU fixtures — no model, no GPU. Mirrors tests/test_grm_runtime_lifecycle
# .py's FakeArena/FakeModel convention (same repo, same pattern).
# ============================================================================

class FakeSelfAttn:
    def __init__(self):
        self.live_shift = 128
        self.inject_kv = None
        self.graft_seats = 0


class FakeLayer:
    def __init__(self):
        self.self_attn = FakeSelfAttn()


class FakeConfig:
    num_layers = 4
    hidden_dim = 256
    kv_lora_rank = 64
    qk_rope_head_dim = 16


class FakeModel:
    config = FakeConfig()

    def __init__(self):
        self.layers = [FakeLayer() for _ in range(self.config.num_layers)]


class FakeArena:
    PAYLOAD = (("c", 1), ("kpe", 2))
    VALS_PER_TOK_LAYER = 96
    ENABLE_ERA_FOLDING = True

    def __init__(self, model, encode, decode, route_layer=2, **_):
        self.m = model
        self.encode = encode
        self.decode = decode
        self.route_layer = route_layer
        self.live_segs = []
        self.grafts = []
        self.node_loader = None
        self.caches, self.pos = None, 0
        self.cur_mounts, self.cur_mount_n = [], 0
        self.page_ins = 0

    def _clear_transients(self):
        pass

    def _bump_cuda_gqa_epoch(self):
        self._cuda_gqa_epoch = getattr(self, "_cuda_gqa_epoch", 0) + 1

    def _ensure_h(self, idxs):
        for i in idxs:
            g = self.grafts[i]
            if g.get("h") is None:
                g["h"] = [{"c": np.zeros((1, 4), dtype=np.float32),
                          "kpe": np.zeros((1, 1, 4), dtype=np.float32)}
                         for _ in self.m.layers]

    def _set_inject(self, att, blk):
        att.inject_kv = (blk["c"], blk["kpe"])
        att.graft_seats = 1

    @staticmethod
    def _rare_tokens(text):
        return {w.lower() for w in text.split() if any(c.isdigit() for c in w)}

    def deposit(self, text):
        idx = len(self.grafts)
        self.grafts.append({
            "h": None,
            "cent": np.full((64,), float(idx), dtype=np.float32),
            "ntok": len(self.encode(text)),
            "text": text,
        })
        return idx

    def feed(self, turn_text):
        idx = self.deposit(turn_text)
        self.grafts[idx]["kind"] = "turn"
        self.grafts[idx]["h"] = [
            {"c": np.zeros((1, 4), dtype=np.float32),
             "kpe": np.zeros((1, 1, 4), dtype=np.float32)}
            for _ in self.m.layers]
        self.live_segs.append((idx, self.grafts[idx]["ntok"]))

    def step(self, user_text, ngen=64, max_trips=2):
        answer = f"Recorded {user_text}"
        self.feed(f"User: {user_text}\nAssistant: {answer}\n")
        return answer, {"ngen": ngen, "max_trips": max_trips, "mounts": []}

    def pack_node(self, h):
        return {"payload_id": np.asarray([0], dtype=np.int64)}

    def unpack_node(self, z):
        return {"id": 0}

    def pack_index(self):
        cents = [g["cent"] for g in self.grafts]
        return {"cents": np.stack(cents) if cents else np.zeros((0, 64),
                                                                np.float32)}

    def unpack_index(self, z, i):
        return z["cents"][i].astype(np.float32)


def enc(text):
    return text.split()


def dec(ids):
    return " ".join(str(i) for i in ids)


def make_repo(tmp_path, **kwargs):
    return GraftRepository(FakeModel(), enc, dec, str(tmp_path),
                           autosave=False, arena_cls=FakeArena,
                           route_layer=2, **kwargs)


# ============================================================================
# 1. Rubric parsing
# ============================================================================

def test_parse_s2_score_accepts_valid_digit_first():
    assert GraftRepository._parse_s2_score(" 3 — critical fact") == 3
    assert GraftRepository._parse_s2_score("0 (throwaway)") == 0
    assert GraftRepository._parse_s2_score("2 useful fact") == 2


def test_parse_s2_score_rejects_garbage():
    assert GraftRepository._parse_s2_score("Sure, I can help with that!") is None
    assert GraftRepository._parse_s2_score("") is None
    assert GraftRepository._parse_s2_score("nine") is None


def test_parse_s2_score_rejects_out_of_range_digit():
    # rubric is 0-3; a stray "5" in the leading position must not parse
    assert GraftRepository._parse_s2_score("5 not on the rubric") is None


def test_parse_s2_score_requires_leading_position():
    # a valid digit that isn't in the fixed leading position is not a
    # looser read of a valid answer — QC law: strict single position
    assert GraftRepository._parse_s2_score(
        "I think this is worth a 2 out of 3") is None


# ============================================================================
# 2. Retry ladder + double-failure -> None
# ============================================================================

def test_s2_score_node_retries_once_then_succeeds(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    idx = repo.arena.deposit("User: hi\nAssistant: hello\n")
    repo.arena.grafts[idx]["kind"] = "turn"

    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "Sure, let me think about that."   # acknowledgment trap
        return "2 — a useful fact worth keeping"

    monkeypatch.setattr(repo, "_s2_generate", fake_generate)

    score = repo._s2_score_node(idx)

    assert score == 2
    assert len(calls) == 2
    assert calls[0] == S2_SALIENCE_PROMPT
    assert calls[1] == S2_SALIENCE_PROMPT


def test_s2_score_node_double_failure_returns_none(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    idx = repo.arena.deposit("User: hi\nAssistant: hello\n")
    repo.arena.grafts[idx]["kind"] = "turn"

    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "I'll go ahead and rate that for you now."

    monkeypatch.setattr(repo, "_s2_generate", fake_generate)

    score = repo._s2_score_node(idx)

    assert score is None
    assert len(calls) == 2        # one retry maximum, never more


def test_s2_score_node_succeeds_first_try_never_retries(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    idx = repo.arena.deposit("User: hi\nAssistant: hello\n")
    repo.arena.grafts[idx]["kind"] = "turn"

    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return "3 critical, must not be lost"

    monkeypatch.setattr(repo, "_s2_generate", fake_generate)

    score = repo._s2_score_node(idx)

    assert score == 3
    assert len(calls) == 1


# ============================================================================
# 3. Manifest write: ONLY the s2_salience key
# ============================================================================

def test_idle_writes_only_s2_salience_key(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("What's the plan for tomorrow?")
    turn_idx = repo.runtime.last_result.new_nodes[0]

    # pre-seed a sibling key (as WO-1/WO-2 would) to prove S2 leaves it
    # alone — "write ONLY that key; other arms own other keys."
    repo.arena.grafts[turn_idx]["metadata"]["importance"]["s1_mass"] = 0.42

    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: "1 minor detail")

    repo.idle(max_jobs=1)

    importance = repo.arena.grafts[turn_idx]["metadata"]["importance"]
    assert importance["s2_salience"] == 1
    assert importance["s1_mass"] == 0.42        # untouched
    assert set(importance.keys()) == {"s1_mass", "s2_salience"}


def test_manifest_round_trips_s2_salience(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Remember the deploy window is Thursday 2am.")
    turn_idx = repo.runtime.last_result.new_nodes[0]
    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: "3 critical fact")
    repo.idle(max_jobs=1)
    repo.flush_now()

    with open(os.path.join(str(tmp_path), "manifest.json")) as fh:
        man = json.load(fh)

    node = man["nodes"][turn_idx]
    assert node["metadata"]["importance"]["s2_salience"] == 3


def test_double_parse_failure_writes_none_not_a_guess(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Just chatting about the weather.")
    turn_idx = repo.runtime.last_result.new_nodes[0]

    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: "As an AI, I can help rate this.")

    repo.idle(max_jobs=1)

    importance = repo.arena.grafts[turn_idx]["metadata"]["importance"]
    assert "s2_salience" in importance
    assert importance["s2_salience"] is None


# ============================================================================
# 4. Opt-in flag off -> pass never scheduled
# ============================================================================

def test_s2_disabled_by_default(tmp_path):
    repo = make_repo(tmp_path)
    assert repo.s2_salience_enabled is False


def test_s2_disabled_idle_never_calls_generate(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)     # s2_salience_enabled defaults False
    repo.chat("Some turn that would otherwise get scored.")
    turn_idx = repo.runtime.last_result.new_nodes[0]

    called = []
    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: called.append(prompt) or "2 fact")

    repo.idle(max_jobs=1)

    assert called == []
    importance = repo.arena.grafts[turn_idx]["metadata"].get("importance", {})
    assert "s2_salience" not in importance


def test_s2_disabled_pending_queue_stays_empty(tmp_path):
    repo = make_repo(tmp_path)
    repo.chat("Anything at all.")
    assert repo._s2_pending == ()


def test_s2_enabled_queues_pending_after_chat(tmp_path):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Anything at all.")
    turn_idx = repo.runtime.last_result.new_nodes[0]
    assert repo._s2_pending == (turn_idx,)


def test_recall_kind_node_never_queued(tmp_path):
    """Derivative-turn hygiene law (2026-06-11 batch): kind="recall" nodes
    are retrieval-only — excluded from routing and folding — so no
    consumer ever reads a salience score off one. _queue_s2_pending()
    must exclude them the same way, not just "turn"."""
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Anything at all.")
    node_idx = repo.runtime.last_result.new_nodes[0]
    repo.arena.grafts[node_idx]["kind"] = "recall"

    repo._queue_s2_pending()

    assert repo._s2_pending == ()


def test_recall_kind_node_never_scored(tmp_path, monkeypatch):
    """End-to-end companion to the queueing test: even if a recall node
    somehow reached idle(), it must not consume a generation call."""
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Anything at all.")
    node_idx = repo.runtime.last_result.new_nodes[0]
    repo.arena.grafts[node_idx]["kind"] = "recall"
    repo._queue_s2_pending()

    called = []
    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: called.append(prompt) or "2 fact")

    repo.idle(max_jobs=1)

    assert called == []
    importance = repo.arena.grafts[node_idx]["metadata"].get("importance", {})
    assert "s2_salience" not in importance


# ============================================================================
# 5. Route/deposit hygiene: the pass never deposits, never mutates routing
# ============================================================================

def test_s2_pass_never_deposits_a_node(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("First turn.")
    before_count = len(repo.arena.grafts)

    monkeypatch.setattr(repo, "_s2_generate", lambda prompt: "2 fact")
    repo.idle(max_jobs=1)

    assert len(repo.arena.grafts) == before_count


def test_s2_pass_skips_retired_node(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("A turn that will be retired before idle scores it.")
    turn_idx = repo.runtime.last_result.new_nodes[0]
    repo.arena.grafts[turn_idx]["retired"] = True

    called = []
    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: called.append(prompt) or "2 fact")

    repo.idle(max_jobs=1)

    assert called == []


def test_s2_pending_is_stateless_across_idle_calls(tmp_path, monkeypatch):
    """Plan pattern (mirrors _librarian_jobs): the pending set is scored
    once and cleared, never re-scored on a later idle() with no new turn
    in between."""
    repo = make_repo(tmp_path, s2_salience_enabled=True)
    repo.chat("Only turn.")
    turn_idx = repo.runtime.last_result.new_nodes[0]

    calls = []
    monkeypatch.setattr(repo, "_s2_generate",
                        lambda prompt: calls.append(1) or "2 fact")

    repo.idle(max_jobs=1)
    repo.idle(max_jobs=1)       # no new turn since — must not re-score

    assert len(calls) == 1
    assert repo.arena.grafts[turn_idx]["metadata"]["importance"][
        "s2_salience"] == 2


# ============================================================================
# GPU SMOKE — script-style, gated behind --run-gpu. NOT executed by pytest
# collection (guarded by __main__ + explicit flag) and NOT executed by this
# authoring agent per work order. The lead runs:
#
#   python3 tests/test_grm_importance_salience.py --run-gpu
#
# Loads MiniCPM3 via core.minicpm3_tc.MiniCPM3_TC, drives a real
# GraftRepository exactly as tests/test_graft_librarian.py does, runs a
# short conversation with s2_salience_enabled=True, deferred librarian
# mode, fires idle(), and asserts:
#   - a real integer 0-3 score lands in the scored node's manifest entry
#   - arena node count/kind/retired state is identical before vs after
#     the scoring pass (snapshot via repo._snapshot_state())
#   - live cache shape (caches/pos/live_segs) is restored to what it was
#     immediately before the scoring pass ran
# ============================================================================

def _run_gpu_smoke():
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import shutil
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from tokenizers import Tokenizer as HFTok

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)

    path = "/tmp/graftrepo_s2_smoke"
    shutil.rmtree(path, ignore_errors=True)
    repo = GraftRepository(
        m, lambda t: tok.encode(t).ids, lambda ids: tok.decode(ids), path,
        autosave=False, librarian_mode="deferred",
        sink_text="<conversation>\n", arena_width=256, route_layer=44,
        topk=3, live_turns=2, ephemeral=True, recency_mounts=2,
        s2_salience_enabled=True)

    repo.chat("The deploy window is Thursday at 2am, code DEPLOY-4471.")
    turn_idx = repo.runtime.last_result.new_nodes[0]
    print(f"turn deposited at node {turn_idx}", flush=True)

    before_state = repo._snapshot_state()
    before_live = (repo.arena.caches, repo.arena.pos,
                  list(repo.arena.live_segs))

    repo.idle(max_jobs=1)

    after_state = repo._snapshot_state()
    after_live = (repo.arena.caches, repo.arena.pos,
                 list(repo.arena.live_segs))

    score = repo.arena.grafts[turn_idx]["metadata"]["importance"].get(
        "s2_salience")
    print(f"s2_salience = {score!r}", flush=True)

    assert score is not None, "S2 pass did not produce a score"
    assert isinstance(score, int) and 0 <= score <= 3, (
        f"score out of rubric range: {score!r}")
    # state tuples exclude live cache deliberately (see _state_tuple) —
    # check node bookkeeping and live cache shape separately.
    assert after_state == before_state, "arena node bookkeeping changed"
    assert after_live[1] == before_live[1], "live position counter drifted"
    assert len(after_live[2]) == len(before_live[2]), (
        "live segment count changed")

    report = {"schema": "grm_importance_s2_smoke_v1",
             "turn_idx": turn_idx, "s2_salience": score}
    print(json.dumps(report), flush=True)
    print("DONE", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRM-IMPORTANCE S2 salience-pass harness")
    parser.add_argument("--run-gpu", action="store_true",
                        help="run the GPU smoke (loads MiniCPM3, no "
                             "co-resident model loads — lead-only, per "
                             "work order GPU discipline)")
    args = parser.parse_args()
    if not args.run_gpu:
        print("No --run-gpu flag: nothing to do here. CPU unit tests run "
              "under pytest (`python3 -m pytest "
              "tests/test_grm_importance_salience.py`). To run the GPU "
              "smoke: python3 tests/test_grm_importance_salience.py "
              "--run-gpu", flush=True)
        sys.exit(0)
    _run_gpu_smoke()
