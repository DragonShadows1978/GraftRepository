from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.grm_det1_3_snapshot import (
    ARM_PROTOCOLS,
    SnapshotError,
    capture_arena_snapshot,
    compare_fork_hydration_delta,
    compare_fork_substrate,
    compare_snapshots,
    finalize_snapshot_with_answer,
    load_snapshot,
    load_snapshot_array,
    process_instance_sha256,
    require_snapshot_member_coverage,
    restore_prefill_fork,
    snapshot_member_coverage,
    verify_fork_masks_consumed,
)
from scripts.grm_det1_3_gpu import (
    _fixture_answer_correct,
    _install_lived_nodes,
    _is_refusal,
    compare_with_observer_controls,
)
from scripts.grm_det1_baseline_registry import (
    CROSS_MODEL_SUP_COMPARATOR,
    EXACT_COMPARATOR,
)
from scripts.grm_det1_e2e import baseline_guard_policy
from scripts.grm_det1_common import (
    contains_value,
    normalize_value_text,
    routing_index_digest,
)


class FakeArena:
    PAYLOAD = (("k", 2), ("v", 2))

    def __init__(self, *, pos: int = 3, live: bool = False):
        attentions = []
        layers = []
        for index in range(2):
            att = SimpleNamespace(
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                num_heads_per_kv=2,
                attention_mode="standard",
                sliding_window=4 if index == 0 else None,
                scaling=0.5,
                bulk_bits=8,
                attn_block=16,
                refine_percentile=0.15,
                layer_type="sliding_attention" if index == 0 else "full_attention",
                inject_kv=None,
                graft_seats=4,
                live_shift=6,
                sinks=np.asarray([index + 0.25] * 4, dtype=np.float32),
            )
            attentions.append(att)
            layers.append(SimpleNamespace(self_attn=att))
        config = SimpleNamespace(
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            rope_theta=10000.0,
        )
        self.m = SimpleNamespace(
            layers=layers,
            config=config,
            rope_cos=np.arange(128, dtype=np.float32).reshape(16, 8),
            rope_sin=np.arange(128, dtype=np.float32).reshape(16, 8) / 10,
            _rope_len=16,
        )
        self.dt = "bfloat16"
        self.storage_bits = 8
        self.route_layer = 1
        self.route_backend = "python"
        self.pos = pos
        self.n_sink = 2
        self.live_shift = 6
        self.width = 4
        self.cur_mount_n = 2
        self.cur_mounts = [0]
        self.topk = 3
        self.live_turns = 2
        self.max_live = 32
        self.cache_deposits = True
        self.revision_resolution = True
        self.decisive_admission = True
        self.length_debias = False
        self.live_segs = [(0, 2)] if live else []
        self.last_route_receipt = {"ranking": [0]}
        self.caches = [
            (
                np.full((1, 2, 4, 8), index + 1, dtype=np.float16),
                np.full((1, 2, 4, 8), index + 2, dtype=np.float16),
            )
            for index in range(2)
        ]
        self.sink_h = [
            {
                "k": np.full((1, 2, 2, 8), index + 3, dtype=np.float16),
                "v": np.full((1, 2, 2, 8), index + 4, dtype=np.float16),
            }
            for index in range(2)
        ]
        self.grafts = [{
            "ntok": 2,
            "h": [
                {
                    "k": np.full((1, 2, 2, 8), index + 5, dtype=np.float16),
                    "v": np.full((1, 2, 2, 8), index + 6, dtype=np.float16),
                }
                for index in range(2)
            ],
        }]

    def _route_cand_base(self):
        return [
            index for index, graft in enumerate(self.grafts)
            if not graft.get("retired") and graft.get("kind", "turn") != "recall"
        ]


IDENTITY = {
    "model_revision": "same-revision",
    "model_weights_inventory_sha256": "weights-hash",
    "tokenizer_sha256": "tokenizer-hash",
    "engine_build_sha256": "engine-hash",
    "prompt_template_sha256": "prompt-template-hash",
    "dialect": "gqa",
}


def admission() -> dict:
    return {
        "schema": "grm.det1_3.admission_snapshot.v1",
        "probe_key_sha256": "probe-key-hash",
        "ranking": [0],
        "route_scores": [{"id": 0, "score": 1.0}],
        "identifier_tokens": ["value"],
        "identified_candidates": [0],
        "policy_branch": "exactly_one_identifier_decisive_rank1",
        "rank_plan": [0],
        "ladder_attempts": [
            {"ordinal": 0, "planned": [0], "clean_room": False}],
        "current_attempt_ordinal": 0,
        "current_planned": [0],
        "current_clean_room": False,
        "current_fitted": [0],
        "current_dropped": [],
        "selection_state": "PENDING_AT_PREFILL",
        "final_mounts": [0],
    }


def provenance(arm: str, process: str, *, attempt: int = 0) -> dict:
    value = {
        "schema": "grm.det1_3.capture_provenance.v2",
        "arm": arm,
        "protocol": ARM_PROTOCOLS[arm],
        "protocol_source_sha256": ("a" if arm == "lived" else "b") * 64,
        "run_id": "run-test",
        "order_sha256": "1" * 64,
        "source_amendment_sha256": "2" * 64,
        "registration_sha256": "3" * 64,
        "runtime_frame_sha256": "4" * 64,
        "fixture_sha256": "5" * 64,
        "fixture_id": "correction_then_restatement",
        "probe_id": "harbor_restatement",
        "probe_question_sha256": "6" * 64,
        "capture_boundary": "before_probe_prefill",
        "selection_boundary": "before_generation_and_grounding_selection",
        "probe_driver": "grm_e2e_session.probe_multimount_chat",
        "topk": 3,
        "max_trips": 1,
        "probe_ladder": True,
        "defer_memory": True,
        "attempt_ordinal": attempt,
        "process_pid": 100 if arm == "lived" else 200,
        "process_start_ticks": 1000 if arm == "lived" else 2000,
        "boot_id": "same-boot",
    }
    value["process_instance_sha256"] = process_instance_sha256(
        value["process_pid"], value["process_start_ticks"], value["boot_id"])
    if process == "lived-process" and arm == "replay":
        value["process_pid"] = 100
        value["process_start_ticks"] = 1000
        value["process_instance_sha256"] = process_instance_sha256(
            value["process_pid"], value["process_start_ticks"], value["boot_id"])
    return value


def linked_answer(
    process: str,
    *,
    answer: str = "Nacre-6-Blue",
    mounts: tuple[int, ...] = (0,),
) -> dict:
    refusal = _is_refusal(answer)
    correct = "nacre-6-blue" in answer.casefold() and not refusal
    return {
        "schema": "grm.det1_3.same_process_linked_answer.v1",
        "process_instance_sha256": process,
        "captured_attempt_ordinal": 0,
        "attempt_answer": answer,
        "attempt_mounts": list(mounts),
        "attempt_answer_correct": correct,
        "attempt_refusal": refusal,
        "probe_answer": answer,
        "probe_selected_attempt": 0,
        "probe_mounts": list(mounts),
        "probe_answer_correct": correct,
        "probe_refusal": refusal,
    }


def capture(
    tmp_path: Path,
    name: str,
    *,
    arena: FakeArena | None = None,
    identity: dict | None = None,
    live_ids=None,
    arm: str | None = None,
    process: str | None = None,
    finalize: bool = True,
    admission_plan: dict | None = None,
) -> Path:
    arena = arena or FakeArena()
    arm = arm or ("lived" if "left" in name else "replay")
    process = process or f"{arm}-process"
    capture_provenance = provenance(arm, process)
    path = capture_arena_snapshot(
        arena,
        tmp_path / name,
        label=arm,
        provenance=capture_provenance,
        question="What is the value?",
        prompt_ids=[7, 8, 9],
        admission_plan=admission_plan or admission(),
        live_token_ids=[] if live_ids is None else live_ids,
        sink_text="<sink>",
        sink_token_ids=[1, 2],
        explicit_identity=identity or IDENTITY,
    )
    if finalize:
        finalize_snapshot_with_answer(
            path,
            linked_answer(
                capture_provenance["process_instance_sha256"],
                mounts=tuple(int(value) for value in arena.cur_mounts),
            ),
        )
    return path


def test_identical_complete_snapshots_pass_state_gate_but_need_controls(tmp_path: Path):
    left = capture(tmp_path, "left")
    right = capture(tmp_path, "right")

    comparison = compare_snapshots(left, right)

    assert comparison["status"] == "PASS_STATE_BYTES_EQUAL_PENDING_OBSERVER_CONTROL"
    assert comparison["state_gate_pass"] is True
    assert comparison["gate_pass"] is False
    assert comparison["counts"]["DIVERGENT"] == 0
    manifest = load_snapshot(left)
    assert manifest["complete"] is True
    assert manifest["arrays"]["cache.layer_00.k"]["shape"] == [1, 2, 4, 8]
    assert manifest["arrays"]["mask.layer_00.allowed"]["host_dtype"] == "|u1"


def test_dynamic_scalar_and_array_divergences_are_named(tmp_path: Path):
    left = capture(tmp_path, "left")
    changed = FakeArena(pos=4)
    changed.caches[1][0][0, 0, 0, 0] = np.float16(99)
    right = capture(tmp_path, "right", arena=changed)

    comparison = compare_snapshots(left, right)
    divergent = {row["field"] for row in comparison["rows"] if row["status"] == "DIVERGENT"}

    assert comparison["status"] == "DIVERGENT"
    assert "state.arena.pos" in divergent
    assert "arrays.cache.layer_01.k" in divergent


def test_self_comparison_and_same_process_cannot_false_pass(tmp_path: Path):
    lived = capture(tmp_path, "left")
    self_comparison = compare_snapshots(lived, lived)
    assert self_comparison["status"] == "INVALID_PROVENANCE"
    assert "SELF_COMPARISON_SAME_MANIFEST" in self_comparison["provenance_errors"]

    replay = capture(
        tmp_path, "right", arm="replay", process="lived-process")
    same_process = compare_snapshots(lived, replay)
    assert same_process["status"] == "INVALID_PROVENANCE"
    assert "CAPTURES_NOT_DISTINCT_FRESH_PROCESSES" in same_process[
        "provenance_errors"]


def _write_control(path: Path, arm: str, process: str, *, answer: str) -> Path:
    refusal = _is_refusal(answer)
    value = {
        "schema": "grm.det1_3.same_model_answer.v1",
        "status": "COMPLETE",
        "arm": arm,
        "protocol": ARM_PROTOCOLS[arm],
        "execution_provenance": {
            **provenance(arm, process),
            "execution_mode": "bare_answer_control",
            "process_pid": 300 if arm == "lived" else 400,
            "process_start_ticks": 3000 if arm == "lived" else 4000,
        },
        "answer": answer,
        "mounted_ids": [0],
        "selected_attempt": 0,
        "answer_correct": "nacre-6-blue" in answer.casefold() and not refusal,
        "answer_refusal": refusal,
    }
    value["execution_provenance"]["process_instance_sha256"] = process_instance_sha256(
        value["execution_provenance"]["process_pid"],
        value["execution_provenance"]["process_start_ticks"],
        value["execution_provenance"]["boot_id"],
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_full_gate_requires_matching_independent_observer_controls(tmp_path: Path):
    lived = capture(tmp_path, "left")
    replay = capture(tmp_path, "right")
    lived_control = _write_control(
        tmp_path / "lived_control.json", "lived", "lived-control", answer="Nacre-6-Blue")
    replay_control = _write_control(
        tmp_path / "replay_control.json", "replay", "replay-control", answer="Nacre-6-Blue")

    comparison = compare_with_observer_controls(
        lived, replay, lived_control, replay_control)

    assert comparison["status"] == "PASS_CANONICAL_VALUE_BYTES_EQUAL"
    assert comparison["gate_pass"] is True
    assert comparison["capture_behavioral_nonperturbation_pass"] is True
    assert comparison["race_resume_authorized"] is False


def test_four_matching_refusals_cannot_certify_failed_lived_premise(tmp_path: Path):
    refusal = "I’m sorry, but I can’t help with that."
    lived = capture(tmp_path, "left", finalize=False)
    replay = capture(tmp_path, "right", finalize=False)
    for path in (lived, replay):
        manifest = load_snapshot(path)
        finalize_snapshot_with_answer(
            path,
            linked_answer(
                manifest["provenance"]["process_instance_sha256"],
                answer=refusal,
            ),
        )
    lived_control = _write_control(
        tmp_path / "lived_refusal.json", "lived", "lived-control", answer=refusal)
    replay_control = _write_control(
        tmp_path / "replay_refusal.json", "replay", "replay-control", answer=refusal)

    comparison = compare_with_observer_controls(
        lived, replay, lived_control, replay_control)

    assert comparison["gate_pass"] is False
    assert "LIVED_ANSWERING_PREMISE_NOT_REPRODUCED" in comparison[
        "behavioral_control_errors"]


def test_retained_curly_apostrophe_harbor_refusal_is_classified():
    assert _is_refusal("I’m sorry, but I can’t help with that.") is True


def test_det1_4_value_normalization_accepts_emphasis_and_unicode_hyphens():
    raw = "The current Harbor token value is **Nacre\u20116\u2010Blue**."
    assert normalize_value_text(raw).endswith("nacre-6-blue.")
    assert contains_value(raw, "Nacre-6-Blue") is True
    fixture = {"probes": [{
        "expected_values": ["nacre-6-blue"],
        "stale_values": ["morrow-5-red"],
        "wrong_fact_values": ["opal-7-green"],
    }]}
    assert _fixture_answer_correct(raw, fixture) is True
    assert _fixture_answer_correct(
        "The value is **Morrow\u20115\u2011Red**.", fixture) is False


def test_det1_10_separator_variants_are_value_equivalent():
    """t30's lived control answered the right value with spaces for hyphens."""
    assert contains_value("Cobalt 1 India", "Cobalt-1-India") is True
    assert contains_value("Cobalt-1-India", "Cobalt-1-India") is True
    assert contains_value("Cobalt\u20111\u2010India", "Cobalt-1-India") is True
    assert contains_value("**Cobalt 1 India**", "Cobalt-1-India") is True
    assert contains_value("cobalt 1 india", "Cobalt-1-India") is True
    assert contains_value(
        "The current atlas tone value is Cobalt 1 India.",
        "Cobalt-1-India",
    ) is True


def test_det1_10_separator_normalization_still_rejects_wrong_values():
    """Separator glyphs are free; token payload and token count are not."""
    # Wrong token in the middle.
    assert contains_value("Cobalt-2-India", "Cobalt-1-India") is False
    assert contains_value("Cobalt 2 India", "Cobalt-1-India") is False
    # Missing token entirely.
    assert contains_value("Cobalt India", "Cobalt-1-India") is False
    assert contains_value("Cobalt-India", "Cobalt-1-India") is False
    # Truncated, reordered, and boundary-straddling forms.
    assert contains_value("Cobalt-1", "Cobalt-1-India") is False
    assert contains_value("India-1-Cobalt", "Cobalt-1-India") is False
    assert contains_value("Cobalt 1 Indiana", "Cobalt-1-India") is False
    assert contains_value("XCobalt-1-India", "Cobalt-1-India") is False
    # A wholly different value must never be rescued by normalization.
    assert contains_value(
        "The Falcon registry value is 42.", "Vortex-3-Sierra") is False


def test_det1_10_comparator_is_consistent_across_det_call_sites():
    """Served controls, registry lawfulness, and baseline classification agree."""
    from scripts.grm_det1_7_registry import _contains_value as registry_contains
    from scripts.grm_det1_baseline_registry import (
        _contains_value as baseline_contains,
    )

    cases = [
        ("Cobalt 1 India", "Cobalt-1-India", True),
        ("Cobalt\u20111\u2011India", "Cobalt-1-India", True),
        ("**Cobalt 1 India**", "Cobalt-1-India", True),
        ("Cobalt-2-India", "Cobalt-1-India", False),
        ("Cobalt India", "Cobalt-1-India", False),
        ("Cobalt 1 Indiana", "Cobalt-1-India", False),
    ]
    for text, value, expected in cases:
        assert contains_value(text, value) is expected, (text, value)
        assert registry_contains(text, value) is expected, (text, value)
        assert baseline_contains(text, value) is expected, (text, value)


def test_lived_install_accepts_non_ephemeral_feed_none_return():
    class Arena:
        def __init__(self):
            self.grafts = []
            self.bumped = False

        @staticmethod
        def encode(text):
            return list(range(len(text.split())))

        def feed(self, text, deposit=True):
            assert deposit is True
            self.grafts.append({"text": text, "ntok": len(self.encode(text))})
            return None

        def _bump_cuda_gqa_epoch(self):
            self.bumped = True

    arena = Arena()
    synced = []
    repo = SimpleNamespace(
        arena=arena, _native_sync_node=lambda index: synced.append(index))
    e2e = SimpleNamespace(
        harmony_turn=lambda user, assistant: f"{user}|{assistant}")
    fixture = {"nodes": [
        {
            "node_id": "a",
            "text": "User: old\nAssistant: noted",
            "supersedes": [],
        },
        {
            "node_id": "b",
            "text": "User: new\nAssistant: updated",
            "supersedes": ["a"],
        },
    ]}

    node_to_idx, ledgers = _install_lived_nodes(repo, e2e, fixture)

    assert node_to_idx == {"a": 0, "b": 1}
    assert set(ledgers) == {0, 1}
    assert arena.grafts[1]["metadata"]["supersedes"] == [0]
    assert arena.grafts[0]["metadata"]["superseded_by"] == [1]
    assert synced == [0, 1]
    assert arena.bumped is True


def test_cross_model_frame_fails_closed_before_dynamic_byte_claim(tmp_path: Path):
    left = capture(tmp_path, "left")
    right = capture(
        tmp_path,
        "right",
        identity={**IDENTITY, "model_revision": "different-model"},
    )

    comparison = compare_snapshots(left, right)

    assert comparison["status"] == "INCOMPATIBLE_FRAME"
    assert comparison["gate_pass"] is False
    assert comparison["counts"]["NOT_COMPARABLE_FRAME"] > 0
    assert any(
        row["field"] == "identity.model_revision"
        and row["status"] == "DIVERGENT"
        for row in comparison["rows"]
    )


def test_phase_mismatch_and_empty_admission_cannot_false_pass(tmp_path: Path):
    left = capture(tmp_path, "left")
    arena = FakeArena()
    right = capture_arena_snapshot(
        arena,
        tmp_path / "right",
        label="replay",
        provenance=provenance("replay", "replay-process"),
        phase="during_decode",
        question="What is the value?",
        prompt_ids=[7, 8, 9],
        admission_plan={},
        live_token_ids=[],
        sink_text="<sink>",
        sink_token_ids=[1, 2],
        explicit_identity=IDENTITY,
    )

    comparison = compare_snapshots(left, right)
    assert load_snapshot(right)["complete"] is False
    assert comparison["gate_pass"] is False
    assert any(
        row["field"] == "metadata.phase" and row["status"] == "DIVERGENT"
        for row in comparison["rows"]
    )


def test_missing_live_token_ledger_marks_snapshot_incomplete(tmp_path: Path):
    arena = FakeArena(live=True)
    path = capture_arena_snapshot(
        arena,
        tmp_path / "incomplete",
        label="lived",
        provenance=provenance("lived", "lived-process"),
        question="What is the value?",
        prompt_ids=[7],
        admission_plan=admission(),
        live_token_ids=None,
        sink_text="<sink>",
        sink_token_ids=[1, 2],
        explicit_identity=IDENTITY,
    )

    manifest = load_snapshot(path)
    assert manifest["complete"] is False
    assert any(
        value["field"] == "tokens.live_ids"
        for value in manifest["availability"]["required_unavailable_fields"]
    )


def test_fork_arm_requires_the_registered_fork_protocol(tmp_path: Path):
    wrong = provenance("fork", "fork-process")
    wrong["protocol"] = ARM_PROTOCOLS["lived"]
    wrong_path = capture_arena_snapshot(
        FakeArena(),
        tmp_path / "wrong_fork_protocol",
        label="fork",
        provenance=wrong,
        question="What is the value?",
        prompt_ids=[7, 8, 9],
        admission_plan=admission(),
        live_token_ids=[],
        sink_text="<sink>",
        sink_token_ids=[1, 2],
        explicit_identity=IDENTITY,
    )
    correct_path = capture_arena_snapshot(
        FakeArena(),
        tmp_path / "correct_fork_protocol",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=[7, 8, 9],
        admission_plan=admission(),
        live_token_ids=[],
        sink_text="<sink>",
        sink_token_ids=[1, 2],
        explicit_identity=IDENTITY,
    )

    wrong_snapshot = load_snapshot(wrong_path)
    correct_snapshot = load_snapshot(correct_path)
    assert wrong_snapshot["complete"] is False
    assert [
        row["field"]
        for row in wrong_snapshot["availability"]["required_unavailable_fields"]
    ] == ["provenance.registered_arm_protocol"]
    assert correct_snapshot["complete"] is True


def test_blob_tamper_is_rejected(tmp_path: Path):
    manifest_path = capture(tmp_path, "snapshot")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(iter(manifest["arrays"].values()))
    blob = manifest_path.parent / record["blob"]
    blob.write_bytes(blob.read_bytes() + b"tamper")

    with pytest.raises(SnapshotError, match="size mismatch"):
        load_snapshot(manifest_path)


def _declared_synthesis_admission(
    *, fitted: tuple[int, ...] = (0, 1), dropped: tuple[int, ...] = (),
) -> dict:
    value = admission()
    value.update({
        "ranking": [0, 1],
        "route_scores": [
            {"id": 0, "score": 1.0},
            {"id": 1, "score": 0.9},
        ],
        "identifier_tokens": ["value", "echo"],
        "identified_candidates": [0, 1],
        "policy_branch": "declared_synthesis_identified_set",
        "rank_plan": [0, 1],
        "ladder_attempts": [{
            "ordinal": 0,
            "planned": [0, 1],
            "clean_room": False,
        }],
        "current_planned": [0, 1],
        "current_fitted": list(fitted),
        "current_dropped": list(dropped),
        "final_mounts": list(fitted),
    })
    return value


def _multi_mount_arena(substrate: str) -> FakeArena:
    arena = FakeArena()
    arena.width = 8
    arena.cur_mounts = [0, 1]
    arena.cur_mount_n = 3
    arena.last_route_receipt = {"ranking": [0, 1]}
    arena.grafts = []
    for graft_id, ntok in ((0, 1), (1, 2)):
        layers = []
        for layer_index in range(2):
            base = np.arange(16 * ntok, dtype=np.float16).reshape(
                1, 2, ntok, 8)
            layers.append({
                "k": base + np.float16(100 * graft_id + 10 * layer_index),
                "v": base + np.float16(200 + 100 * graft_id + 10 * layer_index),
            })
        arena.grafts.append({
            "ntok": ntok,
            "h": layers,
            "text": f"member-{graft_id}",
        })
    if substrate == "injection":
        arena.caches = None
    for layer_index, layer in enumerate(arena.m.layers):
        active = np.arange(80, dtype=np.float16).reshape(1, 2, 5, 8)
        active = active + np.float16(10 * layer_index)
        layer.self_attn.graft_seats = 5
        if substrate == "cache":
            arena.caches[layer_index] = (
                active.copy(), active + np.float16(500))
            layer.self_attn.inject_kv = None
        else:
            layer.self_attn.inject_kv = (
                active.copy(), active + np.float16(500), 1.0)
    return arena


def _declared_single_mount_arena(substrate: str) -> FakeArena:
    arena = _multi_mount_arena(substrate)
    arena.cur_mounts = [0]
    arena.cur_mount_n = 1
    for layer_index, layer in enumerate(arena.m.layers):
        if substrate == "cache":
            cache = arena.caches[layer_index]
            arena.caches[layer_index] = (
                np.ascontiguousarray(cache[0][:, :, :3, :]),
                np.ascontiguousarray(cache[1][:, :, :3, :]),
            )
        else:
            injection = layer.self_attn.inject_kv
            layer.self_attn.inject_kv = (
                np.ascontiguousarray(injection[0][:, :, :3, :]),
                np.ascontiguousarray(injection[1][:, :, :3, :]),
                injection[2],
            )
        layer.self_attn.graft_seats = 3
    return arena


def test_declared_synthesis_budget_drop_still_captures_every_declared_member(
    tmp_path: Path,
):
    arena = _declared_single_mount_arena("cache")
    path = capture(
        tmp_path,
        "declared_synthesis_fitted_single",
        arena=arena,
        arm="lived",
        admission_plan=_declared_synthesis_admission(
            fitted=(0,), dropped=(1,)),
    )

    receipt = require_snapshot_member_coverage(
        load_snapshot(path), "declared synthesis fitted source")

    assert receipt["gate_pass"] is True
    assert receipt["rank_plan_members"] == [0, 1]
    assert receipt["current_fitted_members"] == [0]
    assert receipt["current_dropped_members"] == [1]
    assert receipt["mounted_members"] == [0]
    assert receipt["snapshot_required_members"] == [0, 1]
    assert receipt["declared_snapshot_only_members"] == [1]
    assert receipt["covered_members"] == [0, 1]
    assert receipt["planned_not_mounted_members"] == [1]


def test_snapshot_member_coverage_rejects_contradictory_drop_ledger(
    tmp_path: Path,
):
    path = capture(
        tmp_path,
        "declared_synthesis_bad_drop_ledger",
        arena=_declared_single_mount_arena("cache"),
        arm="lived",
        admission_plan=_declared_synthesis_admission(
            fitted=(0,), dropped=()),
    )

    receipt = snapshot_member_coverage(load_snapshot(path))

    assert receipt["snapshot_complete"] is True
    assert receipt["covered_members"] == [0, 1]
    assert receipt["planned_not_mounted_members"] == [1]
    assert receipt["current_dropped_members"] == []
    assert receipt["plan_fit_consistent"] is False
    assert receipt["gate_pass"] is False


@pytest.mark.parametrize("substrate", ["cache", "injection"])
def test_declared_single_physical_mount_delta_retains_other_member_snapshot(
    tmp_path: Path, substrate: str,
):
    source = capture(
        tmp_path,
        f"declared_single_physical_source_{substrate}",
        arena=_declared_single_mount_arena(substrate),
        arm="lived",
        admission_plan=_declared_synthesis_admission(
            fitted=(0,), dropped=(1,)),
    )
    target = _multi_mount_arena(substrate)
    restore = restore_prefill_fork(
        target,
        source,
        withheld_mounts=[0],
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )
    fork = capture_arena_snapshot(
        target,
        tmp_path / f"declared_single_physical_fork_{substrate}",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=restore["prompt_ids"],
        admission_plan=restore["admission_state"],
        live_token_ids=restore["live_token_ids"],
        sink_text="<sink>",
        sink_token_ids=restore["sink_token_ids"],
        explicit_identity=IDENTITY,
    )

    delta = compare_fork_hydration_delta(
        source, fork, withheld_mounts=[0])
    member_delta = delta["member_snapshot_coverage"]

    assert delta["status"] == (
        "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES")
    assert delta["gate_pass"] is True
    assert delta["source_mounts"] == [0]
    assert delta["fork_mounts"] == []
    assert member_delta["source_snapshot_members"] == [0, 1]
    assert member_delta["expected_fork_snapshot_members"] == [1]
    assert member_delta["lived"]["covered_members"] == [0, 1]
    assert member_delta["fork"]["covered_members"] == [1]
    assert member_delta["retained_member_payload_field_count"] == 4
    assert member_delta["all_other_member_payload_bytes_exact"] is True


def test_missing_one_mounted_member_payload_fails_coverage_closed(
    tmp_path: Path,
):
    arena = _multi_mount_arena("cache")
    del arena.grafts[1]["h"][1]["v"]
    path = capture(
        tmp_path,
        "declared_synthesis_missing_member",
        arena=arena,
        arm="lived",
        finalize=False,
        admission_plan=_declared_synthesis_admission(),
    )

    receipt = snapshot_member_coverage(load_snapshot(path))

    assert receipt["gate_pass"] is False
    assert receipt["mounted_members"] == [0, 1]
    assert receipt["covered_members"] == [0]
    assert receipt["missing_member_payload_fields"] == [
        "mounted_graft.1.layer_01.v"]
    with pytest.raises(SnapshotError, match="snapshot-member coverage"):
        require_snapshot_member_coverage(
            load_snapshot(path), "incomplete declared synthesis")


def _cpu_fork_tensor(array: np.ndarray, _source_dtype: str) -> np.ndarray:
    return np.ascontiguousarray(array.copy())


def test_zero_intervention_fork_restores_canonical_snapshot_values(tmp_path: Path):
    source = capture(tmp_path, "source", arena=FakeArena(), arm="lived")
    target = FakeArena(pos=99)
    target.m.rope_cos.fill(-1)
    target.m.rope_sin.fill(-2)
    target.sink_h[0]["k"].fill(-3)
    target.caches[0][0].fill(-4)

    receipt = restore_prefill_fork(
        target,
        source,
        target_identity=IDENTITY,
        expected_frame_sha256=load_snapshot(source)["identity"]["frame_sha256"],
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )

    assert receipt["intervention"] == "ZERO"
    assert receipt["zero_intervention_no_deltas"] is True
    assert receipt["field_deltas"] == []
    assert receipt["prompt_ids"] == [7, 8, 9]
    assert target.pos == 3
    assert target.cur_mounts == [0]
    assert target.cur_mount_n == 2
    np.testing.assert_array_equal(
        target.caches[0][0], load_snapshot_array(source, "cache.layer_00.k"))
    np.testing.assert_array_equal(
        target.m.rope_cos, load_snapshot_array(source, "model.rope_cos"))
    np.testing.assert_array_equal(
        target.m.layers[0].self_attn._det1_fork_allowed_mask,
        load_snapshot_array(source, "mask.layer_00.allowed"),
    )
    assert all(
        row["bytes_equal_expected_fork_value"]
        for row in receipt["array_verification"].values()
    )

    fork = capture(tmp_path, "fork", arena=target)
    comparison = compare_fork_substrate(source, fork)
    assert comparison["status"] == "PASS_EXACT_CANONICAL_VALUE_BYTES"
    assert comparison["gate_pass"] is True
    assert comparison["counts"] == {"EQUAL": len(comparison["rows"])}


@pytest.mark.parametrize("substrate", ["cache", "injection"])
def test_planted_miss_fork_removes_only_mounted_seat_ranges(
    tmp_path: Path, substrate: str,
):
    source_arena = FakeArena()
    for layer_index, layer in enumerate(source_arena.m.layers):
        k = np.arange(64, dtype=np.float16).reshape(1, 2, 4, 8) + layer_index
        v = k + np.float16(100)
        layer.self_attn.graft_seats = 4
        if substrate == "cache":
            source_arena.caches[layer_index] = (k, v)
            layer.self_attn.inject_kv = None
        else:
            source_arena.caches = None
            layer.self_attn.inject_kv = (k, v, 1.0)
    source = capture(
        tmp_path, f"source_{substrate}", arena=source_arena, arm="lived")
    target = FakeArena()
    if substrate == "injection":
        target.caches = None

    receipt = restore_prefill_fork(
        target,
        source,
        withheld_mounts=[0],
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )

    assert receipt["intervention"] == "REGISTERED_PLANTED_MISS_WITHHOLDING"
    assert receipt["removed_seat_ranges"] == [[2, 4]]
    assert receipt["fork_mounts"] == []
    assert target.cur_mounts == []
    assert target.cur_mount_n == 0
    assert target.m.layers[0].self_attn.graft_seats == 2
    restored_k = (
        target.caches[0][0]
        if substrate == "cache"
        else target.m.layers[0].self_attn.inject_kv[0]
    )
    source_field = (
        "cache.layer_00.k"
        if substrate == "cache"
        else "injection.layer_00.k"
    )
    np.testing.assert_array_equal(
        restored_k,
        load_snapshot_array(source, source_field)[:, :, :2, :],
    )
    assert target.m.layers[0].self_attn._det1_fork_allowed_mask.shape == (
        1, 1, 3, 5)
    assert receipt["admission_state"]["rank_plan"] == []
    assert receipt["admission_state"]["current_planned"] == []
    assert receipt["admission_state"]["final_mounts"] == []
    assert receipt["zero_intervention_no_deltas"] is False
    assert receipt["field_deltas"]
    assert {
        row["reason"] for row in receipt["field_deltas"]
    } == {"registered_planted_miss_withholding"}

    fork = capture_arena_snapshot(
        target,
        tmp_path / f"fork_{substrate}",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=receipt["prompt_ids"],
        admission_plan=receipt["admission_state"],
        live_token_ids=receipt["live_token_ids"],
        sink_text="<sink>",
        sink_token_ids=receipt["sink_token_ids"],
        explicit_identity=IDENTITY,
    )
    delta = compare_fork_hydration_delta(
        source, fork, withheld_mounts=[0])
    assert delta["status"] == (
        "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES")
    assert delta["gate_pass"] is True
    assert delta["target_absence"]["gate_pass"] is True
    assert delta["exact_divergence_set"] is True
    assert delta["non_delta_fields_equal"] is True
    assert delta["unexpected_divergent_fields"] == []
    assert delta["missing_expected_divergent_fields"] == []
    assert delta["expected_fork_value_mismatches"] == []
    assert delta["expected_divergent_fields"] == delta[
        "observed_non_equal_fields"]
    assert delta["strict_zero_comparator"]["status"] == "DIVERGENT"
    assert delta["strict_zero_comparator"]["gate_pass"] is False
    expected_counts = (
        {"DIVERGENT": 21, "EQUAL": 51, "MISSING_FORK": 4}
        if substrate == "cache" else
        {"DIVERGENT": 20, "EQUAL": 54, "MISSING_FORK": 4}
    )
    assert delta["strict_zero_comparator"]["counts"] == expected_counts


@pytest.mark.parametrize("substrate", ["cache", "injection"])
@pytest.mark.parametrize(
    ("withheld", "kept", "seat_range"),
    [(0, 1, [2, 3]), (1, 0, [3, 5])],
)
def test_declared_synthesis_delta_withholds_one_member_and_preserves_other(
    tmp_path: Path,
    substrate: str,
    withheld: int,
    kept: int,
    seat_range: list[int],
):
    source_arena = _multi_mount_arena(substrate)
    source = capture(
        tmp_path,
        f"declared_synthesis_source_{substrate}",
        arena=source_arena,
        arm="lived",
        admission_plan=_declared_synthesis_admission(),
    )
    source_snapshot = load_snapshot(source)
    source_coverage = require_snapshot_member_coverage(
        source_snapshot, "declared synthesis source")
    target = _multi_mount_arena(substrate)

    restore = restore_prefill_fork(
        target,
        source,
        withheld_mounts=[withheld],
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )
    fork = capture_arena_snapshot(
        target,
        tmp_path / f"declared_synthesis_fork_{substrate}",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=restore["prompt_ids"],
        admission_plan=restore["admission_state"],
        live_token_ids=restore["live_token_ids"],
        sink_text="<sink>",
        sink_token_ids=restore["sink_token_ids"],
        explicit_identity=IDENTITY,
    )
    fork_snapshot = load_snapshot(fork)
    fork_coverage = require_snapshot_member_coverage(
        fork_snapshot, "declared synthesis target-only fork")
    delta = compare_fork_hydration_delta(
        source, fork, withheld_mounts=[withheld])

    assert source_coverage["mounted_members"] == [0, 1]
    assert source_coverage["covered_members"] == [0, 1]
    assert fork_coverage["mounted_members"] == [kept]
    assert fork_coverage["covered_members"] == [kept]
    assert delta["source_mounted_aliases"] == [withheld]
    assert delta["source_mounts"] == [0, 1]
    assert delta["fork_mounts"] == [kept]
    assert delta["seat_ranges"] == [{
        "graft_id": withheld,
        "range": seat_range,
    }]
    assert delta["target_absence"]["gate_pass"] is True
    assert delta["retained_arrays_exact"] is True
    assert delta["non_delta_fields_equal"] is True
    assert delta["expected_fork_value_mismatches"] == []
    member_delta = delta["member_snapshot_coverage"]
    assert member_delta["gate_pass"] is True
    assert member_delta["selected_member_payload_absent"] is True
    assert member_delta["all_other_member_payload_bytes_exact"] is True
    assert member_delta["lived"]["covered_members"] == [0, 1]
    assert member_delta["fork"]["covered_members"] == [kept]
    for field, record in source_snapshot["arrays"].items():
        if field.startswith(f"mounted_graft.{withheld}."):
            assert field not in fork_snapshot["arrays"]
        if field.startswith(f"mounted_graft.{kept}."):
            assert fork_snapshot["arrays"][field]["sha256"] == record["sha256"]


def test_declared_synthesis_delta_rejects_retained_member_payload_mutation(
    tmp_path: Path,
):
    source = capture(
        tmp_path,
        "declared_synthesis_survivor_source",
        arena=_multi_mount_arena("cache"),
        arm="lived",
        admission_plan=_declared_synthesis_admission(),
    )
    target = _multi_mount_arena("cache")
    restore = restore_prefill_fork(
        target,
        source,
        withheld_mounts=[0],
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )
    target.grafts[1]["h"][0]["k"][0, 0, 0, 0] += np.float16(1)
    fork = capture_arena_snapshot(
        target,
        tmp_path / "declared_synthesis_survivor_mutated",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=restore["prompt_ids"],
        admission_plan=restore["admission_state"],
        live_token_ids=restore["live_token_ids"],
        sink_text="<sink>",
        sink_token_ids=restore["sink_token_ids"],
        explicit_identity=IDENTITY,
    )

    delta = compare_fork_hydration_delta(
        source, fork, withheld_mounts=[0])

    assert delta["gate_pass"] is False
    assert delta["retained_arrays_exact"] is False
    assert delta["member_snapshot_coverage"]["gate_pass"] is False
    assert delta["member_snapshot_coverage"][
        "all_other_member_payload_bytes_exact"] is False
    assert "arrays.mounted_graft.1.layer_00.k" in delta[
        "expected_fork_value_mismatches"]


def test_planted_delta_comparator_rejects_an_unrelated_byte_change(tmp_path: Path):
    source = capture(tmp_path, "source", arena=FakeArena(), arm="lived")
    target = FakeArena()
    receipt = restore_prefill_fork(
        target,
        source,
        withheld_mounts=[0],
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )
    target.m.rope_cos[0, 0] += np.float32(1.0)
    fork = capture_arena_snapshot(
        target,
        tmp_path / "fork_unrelated_mutation",
        label="fork",
        provenance=provenance("fork", "fork-process"),
        question="What is the value?",
        prompt_ids=receipt["prompt_ids"],
        admission_plan=receipt["admission_state"],
        live_token_ids=receipt["live_token_ids"],
        sink_text="<sink>",
        sink_token_ids=receipt["sink_token_ids"],
        explicit_identity=IDENTITY,
    )

    delta = compare_fork_hydration_delta(
        source, fork, withheld_mounts=[0])

    assert delta["gate_pass"] is False
    assert delta["non_delta_fields_equal"] is False
    assert "arrays.model.rope_cos" in delta["unexpected_divergent_fields"]
    assert "arrays.model.rope_cos" in delta["expected_fork_value_mismatches"]


def test_planted_miss_fork_rejects_a_model_state_noop(tmp_path: Path):
    source = capture(tmp_path, "source", arena=FakeArena(), arm="lived")
    target = FakeArena()
    target.grafts.append({"ntok": 1, "h": None})

    with pytest.raises(SnapshotError, match="model-state no-op"):
        restore_prefill_fork(
            target,
            source,
            withheld_mounts=[1],
            target_identity=IDENTITY,
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )


def test_planted_miss_fork_rejects_duplicate_or_live_target(tmp_path: Path):
    source = capture(tmp_path, "source", arena=FakeArena(), arm="lived")
    with pytest.raises(SnapshotError, match="duplicates"):
        restore_prefill_fork(
            FakeArena(),
            source,
            withheld_mounts=[0, 0],
            target_identity=IDENTITY,
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )

    live_source = capture(
        tmp_path,
        "live_source",
        arena=FakeArena(live=True),
        arm="lived",
        live_ids=[10, 11],
    )
    with pytest.raises(SnapshotError, match="live seats"):
        restore_prefill_fork(
            FakeArena(live=True),
            live_source,
            withheld_mounts=[0],
            target_identity=IDENTITY,
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )


def test_fork_restore_rejects_replay_arm_and_wrong_target_frame(tmp_path: Path):
    replay = capture(tmp_path, "replay_source", arena=FakeArena(), arm="replay")
    with pytest.raises(SnapshotError, match="registered lived capture"):
        restore_prefill_fork(
            FakeArena(), replay,
            target_identity=IDENTITY,
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )

    lived = capture(tmp_path, "lived_source", arena=FakeArena(), arm="lived")
    with pytest.raises(SnapshotError, match="arena frame differs"):
        restore_prefill_fork(
            FakeArena(), lived,
            target_identity={**IDENTITY, "model_revision": "wrong-revision"},
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )


def test_fork_detector_mode_requires_same_lived_process_index(tmp_path: Path):
    lived = capture(tmp_path, "lived_source", arena=FakeArena(), arm="lived")
    with pytest.raises(SnapshotError, match="full lived repository/index"):
        restore_prefill_fork(
            FakeArena(), lived,
            target_identity=IDENTITY,
            tensor_factory=_cpu_fork_tensor,
        )


def test_snapshot_mapping_record_substitution_is_rejected(tmp_path: Path):
    lived = capture(tmp_path, "lived_source", arena=FakeArena(), arm="lived")
    mapping = load_snapshot(lived)
    mapping["arrays"]["cache.layer_00.k"]["sha256"] = "0" * 64
    with pytest.raises(SnapshotError, match="differs from its signed manifest"):
        load_snapshot_array(mapping, "cache.layer_00.k")


def test_all_layer_mounted_geometry_is_validated(tmp_path: Path):
    lived = capture(tmp_path, "lived_source", arena=FakeArena(), arm="lived")
    value = json.loads(lived.read_text(encoding="utf-8"))
    value["arrays"]["mounted_graft.0.layer_01.k"]["shape"][2] = 1
    value.pop("manifest_payload_sha256")
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    value["manifest_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    lived.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(SnapshotError, match="inconsistent sequence lengths"):
        restore_prefill_fork(
            FakeArena(), lived,
            target_identity=IDENTITY,
            require_same_process_index=False,
            tensor_factory=_cpu_fork_tensor,
        )


def test_fork_masks_must_be_consumed_after_prefill(tmp_path: Path):
    lived = capture(tmp_path, "lived_source", arena=FakeArena(), arm="lived")
    target = FakeArena()
    restore_prefill_fork(
        target, lived,
        target_identity=IDENTITY,
        require_same_process_index=False,
        tensor_factory=_cpu_fork_tensor,
    )
    with pytest.raises(SnapshotError, match="did not consume"):
        verify_fork_masks_consumed(target)
    for layer in target.m.layers:
        layer.self_attn._det1_fork_allowed_mask = None
        layer.self_attn._det1_fork_mask_consumed = True
    assert verify_fork_masks_consumed(target)["gate_pass"] is True


def test_routing_index_digest_binds_lqr_inputs_but_not_mounted_payload():
    arena = FakeArena()
    arena.grafts[0].update({
        "text": "Harbor value",
        "cent": np.asarray([0.25, 0.75], dtype=np.float32),
        "metadata": {"supersedes": [], "superseded_by": []},
    })
    before = routing_index_digest(arena)
    arena.grafts[0]["h"][0]["k"].fill(99)
    assert routing_index_digest(arena) == before
    arena.grafts[0]["cent"][0] = np.float32(0.5)
    assert routing_index_digest(arena)["projection_sha256"] != before[
        "projection_sha256"]


def test_cross_model_acceptance_is_never_a_purity_stop():
    cross_model = baseline_guard_policy({
        "comparator": CROSS_MODEL_SUP_COMPARATOR,
        "live_match": False,
    })
    same_model = baseline_guard_policy({
        "comparator": EXACT_COMPARATOR,
        "live_match": False,
    })

    assert cross_model == {
        "schema": "grm.det1.baseline_guard_policy.v1",
        "role": "cross_model_task_acceptance_only",
        "enforce_before_same_run_behavioral_differential": False,
        "can_diagnose_instrumentation_leak": False,
        "task_acceptance": False,
    }
    assert same_model["role"] == "same_model_behavioral_baseline"
    assert same_model["enforce_before_same_run_behavioral_differential"] is False
    assert same_model["can_diagnose_instrumentation_leak"] is False
