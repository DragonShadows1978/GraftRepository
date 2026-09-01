"""LSR-P2B — route-receipt persistence in the per-turn memory ledger.

Phase 1 earned the principle "route receipts are not persisted at serving
time".  These tests hold the fix to its stated contract:

  * a receipt is written on EVERY turn -- mutation-free, abstained, and
    ``no_mount_fit`` turns included;
  * ``fit_*`` / ``abstain*`` / ``served_without_plan_head`` pass through
    generically, so P2A's new fields persist with no P2B re-edit;
  * the schema sha is canonical-JSON stable (fixed fixture -> fixed sha);
  * the FROZEN memory-ledger mutation rows are byte-identical with and
    without the route receipt attached.
"""

import json

import pytest

from core.grm_three_pass import (
    MEMORY_LEDGER_SCHEMA,
    ROUTE_RECEIPT_SCHEMA,
    MemoryLedgerBuilder,
    build_route_receipt,
    canonical_json_bytes,
    route_receipt_sha256,
)
from scripts.lsr_p2b_route_receipt_dump import (
    EVIDENCE_FIELDS,
    collect_receipts,
    evidence_block,
    turn_block,
)


class FakeArena:
    def __init__(self, width=96):
        self.grafts = []
        self._s4_turn = 0
        self._cuda_gqa_epoch = 0
        self.width = width
        self.cur_mounts = []
        self.cur_mount_n = 0
        self.last_route_backend = "python"
        self.last_route_receipt = None


class FakeRepository:
    def __init__(self, width=96):
        self.arena = FakeArena(width=width)


def _node(text, *, active=True):
    return {
        "text": text,
        "ntok": len(text.split()),
        "kind": "turn",
        "retired": not active,
        "metadata": {"active": active, "importance": {}, "supersedes": []},
        "provenance": [],
    }


# The lived sup_lumen_head evidence block, copied from the Phase 1
# adjudication artifact (lsr_p1_adjudication_31e5c89453f4aa42.json).
LUMEN_INFO = {
    "mounts": [2],
    "mount_plan": [1, 0],
    "mount_fitted": [1],
    "mount_dropped_for_width": [0],
    "ranking_ids": [1, 0],
    "trip": 0,
    "admission_policy": "A-DEC",
    "admission_policy_branch": "declared_synthesis_identified_set",
    "admission_rank_plan": [1, 0],
    "admission_identified_candidates": [1, 0],
    "admission_identifier_hit_count": 2,
    "admission_route_margin_1_2": 0.5,
    "admission_route_margin_evaluated": True,
    "admission_margin_threshold": 0.1385774091529802,
    "admission_rule_sha256": (
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"),
}
LUMEN_PROFILE = {
    "identifier_tokens": ["lumen", "seal"],
    "rare_identifier_tokens": [],
    "ranking": [1, 0],
    "rank_plan": [1, 0],
    "identified_candidates": [1, 0],
}


def _lumen_receipt(**overrides):
    kwargs = dict(
        session_id="lsr-p2b",
        turn_id="0",
        info=dict(LUMEN_INFO),
        admission_profile=dict(LUMEN_PROFILE),
        route_limit=6,
        candidate_count=2,
        repository_size=2,
        serving_path="fixture",
        turn_kind="probe",
    )
    kwargs.update(overrides)
    return build_route_receipt(**kwargs)


# --------------------------------------------------------------- schema sha

def test_route_receipt_sha_is_canonical_and_stable():
    """Fixed fixture -> fixed sha, and the stamp never covers itself."""
    first = _lumen_receipt()
    second = _lumen_receipt()
    assert first["schema"] == ROUTE_RECEIPT_SCHEMA
    assert first == second
    assert first["receipt_sha256"] == route_receipt_sha256(first)
    # Registered value: this exact fixture hashes here.  A change to the
    # record's shape must be a NEW schema version, not a silent re-stamp.
    #
    # RE-BLESSED once, at the P2A/P2B merge:
    #   ac66373ed5c9cdce08b2c92389177155331350358074f76cd3113eababe28c2f
    #     -> 0610400b35176d14fca9e37fb7b9a8ce2530156f7d7936e6479a44d13de04614
    #   cause: fit.trips[] rows gained a "shuttle" bool so LSR-P2A Ruling 1.2
    #   shuttle rungs are distinguishable from ordinary trips.  Sole pin in
    #   the tree; no persisted artifact carried the old value.
    assert first["receipt_sha256"] == (
        "0610400b35176d14fca9e37fb7b9a8ce2530156f7d7936e6479a44d13de04614")
    # The sha covers the record MINUS the stamp.
    payload = {k: v for k, v in first.items() if k != "receipt_sha256"}
    import hashlib
    assert first["receipt_sha256"] == hashlib.sha256(
        canonical_json_bytes(payload)).hexdigest()
    # A single changed field moves it.
    moved = _lumen_receipt(repository_size=3)
    assert moved["receipt_sha256"] != first["receipt_sha256"]


def test_route_receipt_reproduces_phase1_evidence_block():
    receipt = _lumen_receipt()
    evidence = evidence_block(receipt)
    assert evidence["admission.ranking"] == [1, 0]
    assert evidence["admission.identified_candidates"] == [1, 0]
    assert evidence["admission.identifier_tokens"] == ["lumen", "seal"]
    assert evidence["admission.policy_branch"] == (
        "declared_synthesis_identified_set")
    assert evidence["admission.rank_plan"] == [1, 0]
    assert evidence["admission.current_planned"] == [1, 0]
    assert evidence["admission.current_fitted"] == [1]
    assert evidence["admission.current_dropped"] == [0]
    assert evidence["admission.final_mounts"] == [1]
    assert evidence["admission.rule_sha256"] == (
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91")
    assert evidence["lived_repository_size_at_probe"] == 2
    assert evidence["ranking_length"] == 2
    assert set(EVIDENCE_FIELDS) == set(evidence)


# ------------------------------------------------ written on EVERY turn

def test_receipt_written_on_mutation_free_turn():
    """No mutation at all still produces a route receipt row."""
    repo = FakeRepository()
    builder = MemoryLedgerBuilder(
        repo, session_id="s", turn_id="7",
        request_text="what is the lumen seal value?", output_text="idk")
    builder.attach_route_receipt(_lumen_receipt(turn_id="7"))
    receipt, audit = builder.finalize()
    assert receipt["mutation_count"] == 0
    assert receipt["mutations"] == []
    assert audit["complete"] is True
    assert receipt["route_receipt"]["schema"] == ROUTE_RECEIPT_SCHEMA
    assert receipt["route_receipt"]["turn_id"] == "7"


def test_receipt_written_on_abstained_and_no_mount_fit_turn():
    """Abstention / no_mount_fit is exactly the turn Phase 1 could not read."""
    info = {
        "trip": 0,
        "no_mount_fit": True,
        "mount_plan": [2, 1, 0],
        "mount_fitted": [],
        "mount_dropped_for_width": [2, 1, 0],
        "ranking_ids": [2, 1, 0],
        "abstained": True,
        "abstain_reason": "not_in_memory",
    }
    arena = FakeArena(width=96)
    arena.cur_mount_n = 0
    record = build_route_receipt(
        session_id="s", turn_id="3", info=info, arena=arena,
        route_limit=6, candidate_count=3, serving_path="fixture",
        turn_kind="probe")
    assert record["fit"]["no_mount_fit"] is True
    assert record["fit"]["seated"] == []
    assert record["fit"]["dropped"] == [2, 1, 0]
    assert record["fit"]["info_pass_through"]["abstained"] is True
    assert record["fit"]["info_pass_through"]["abstain_reason"] == (
        "not_in_memory")
    assert record["route"]["ranking_ids"] == [2, 1, 0]

    repo = FakeRepository()
    builder = MemoryLedgerBuilder(
        repo, session_id="s", turn_id="3",
        request_text="q", output_text="")
    builder.attach_route_receipt(record)
    ledger_receipt, audit = builder.finalize()
    assert audit["complete"] is True
    assert ledger_receipt["route_receipt"]["fit"]["no_mount_fit"] is True


def test_generic_fit_and_abstain_passthrough_needs_no_p2b_edit():
    """P2A's future info keys persist by PREFIX, not by an enumerated list."""
    info = {
        "trip": 1,
        # P2A fit-honesty shape (names invented here on purpose: the point is
        # that P2B does not need to know them).
        "fit_honesty_verdict": "planned_head_unseatable",
        "fit_shuttle_trip_triggered": True,
        "fit_degraded_flag": "width_96_long_competitor",
        "abstain": True,
        "abstain_policy": "not_in_memory_v1",
        "served_without_plan_head": True,
        # Not pass-through material: already projected, or unrelated.
        "mounts": [1],
        "resident": 200,
        "driver_topk": 3,
        "unrelated_key": "ignored",
    }
    record = build_route_receipt(
        session_id="s", turn_id="1", info=info, serving_path="fixture")
    through = record["fit"]["info_pass_through"]
    assert through == {
        "fit_honesty_verdict": "planned_head_unseatable",
        "fit_shuttle_trip_triggered": True,
        "fit_degraded_flag": "width_96_long_competitor",
        "abstain": True,
        "abstain_policy": "not_in_memory_v1",
        "served_without_plan_head": True,
    }
    assert "unrelated_key" not in through
    assert "resident" not in through
    assert "driver_topk" not in through


# --------------------------------------- FROZEN mutation rows stay frozen

def _run_deposit_ledger(repo, *, route_receipt):
    builder = MemoryLedgerBuilder(
        repo, session_id="session-a", turn_id="0",
        request_text="remember alpha", output_text="ack")
    builder.attach_route_receipt(route_receipt)

    def deposit():
        repo.arena.grafts.append(_node("alpha"))
        repo.arena._cuda_gqa_epoch += 1
        repo.arena._s4_turn += 1

    builder.record_operation(
        "deposit", reason_code="test_deposit", reason_detail="fixture",
        source_text="alpha", operation=deposit)
    return builder.finalize()


def test_mutation_rows_byte_identical_with_and_without_route_receipt():
    plain, plain_audit = _run_deposit_ledger(FakeRepository(),
                                             route_receipt=None)
    stamped, stamped_audit = _run_deposit_ledger(
        FakeRepository(), route_receipt=_lumen_receipt())

    assert canonical_json_bytes(plain["mutations"]) == canonical_json_bytes(
        stamped["mutations"])
    assert plain["mutation_count"] == stamped["mutation_count"]
    assert canonical_json_bytes(plain["provenance"]) == canonical_json_bytes(
        stamped["provenance"])
    assert canonical_json_bytes(plain_audit) == canonical_json_bytes(
        stamped_audit)
    # Old consumers see the frozen key set unchanged when nothing is attached.
    assert set(plain) == {
        "schema", "session_id", "turn_id", "turn_pipeline", "pass",
        "provenance", "mutations", "mutation_count",
    }
    assert plain["schema"] == MEMORY_LEDGER_SCHEMA
    # The receipt is a SIDECAR: exactly one added key, nothing rewritten.
    assert set(stamped) - set(plain) == {"route_receipt"}
    assert stamped["schema"] == MEMORY_LEDGER_SCHEMA


def test_attach_rejects_a_foreign_schema():
    repo = FakeRepository()
    builder = MemoryLedgerBuilder(
        repo, session_id="s", turn_id="0", request_text="q", output_text="a")
    with pytest.raises(ValueError, match="grm.route_receipt.v1"):
        builder.attach_route_receipt({"schema": "grm.memory_ledger.turn.v1"})


# ------------------------------------------------------------- dump script

def test_dump_script_reads_receipts_from_a_session_tree(tmp_path):
    ledger_dir = tmp_path / "memory_ledger"
    ledger_dir.mkdir()
    for turn in range(3):
        repo = FakeRepository()
        builder = MemoryLedgerBuilder(
            repo, session_id="sess", turn_id=str(turn),
            request_text="q%d" % turn, output_text="a%d" % turn)
        builder.attach_route_receipt(_lumen_receipt(
            turn_id=str(turn), repository_size=turn + 1))
        receipt, _audit = builder.finalize()
        (ledger_dir / ("turn_%04d.json" % turn)).write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    # An instrumentation row carrying the receipt is picked up too, and the
    # duplicate (same sha) is not double-counted.
    (tmp_path / "instrumentation.jsonl").write_text(
        json.dumps({"turn": 0, "route_receipt": _lumen_receipt(turn_id="0",
                                                               repository_size=1)})
        + "\n", encoding="utf-8")

    receipts = collect_receipts(str(tmp_path))
    assert [r["turn_id"] for r in receipts] == ["0", "1", "2"]
    blocks = [turn_block(r) for r in receipts]
    assert all(b["receipt_sha256_ok"] for b in blocks)
    assert blocks[0]["evidence"]["admission.final_mounts"] == [1]


def test_dump_script_reads_a_runtime_info_receipt(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"info": {"route_receipt_record": _lumen_receipt()}}) + "\n",
        encoding="utf-8")
    receipts = collect_receipts(str(path))
    assert len(receipts) == 1
    assert receipts[0]["schema"] == ROUTE_RECEIPT_SCHEMA


# ------------------------------------------------------------- G2 replay

def test_g2_replay_reproduces_phase1_evidence_when_receipts_are_present():
    """LSR-P2B G2, as a regression: 8/8 probes, field for field.

    The DET1.5 shard tree and the Phase 1 adjudication are large read-only
    campaign artifacts that need not live in every checkout; when they are
    absent the gate is skipped rather than silently passing.
    """
    import os

    from scripts.lsr_p2b_g2_replay import main as g2_main

    roots = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "/mnt/ForgeRealm/GraftRepository",
    ]
    adjudication = None
    det_root = None
    for root in roots:
        candidate = os.path.join(
            root, "artifacts", "lsr_p1",
            "lsr_p1_adjudication_31e5c89453f4aa42.json")
        shards = os.path.join(
            root, "artifacts", "grm_det1", "run_20260831T160525Z_2")
        if os.path.isfile(candidate) and os.path.isdir(shards):
            adjudication, det_root = candidate, root
            break
    if adjudication is None:
        pytest.skip("DET1.5 lived receipts / Phase 1 adjudication not present")

    rc = g2_main([
        "--det-root", det_root,
        "--p1-adjudication", adjudication,
    ])
    assert rc == 0


# ------------------------------------------------------- production path

class _StepArena(FakeArena):
    def __init__(self):
        super().__init__()
        self.grafts = [_node("alpha")]
        self.topk = 3
        self.cur_mounts = [0]
        self.cur_mount_n = 1

    def step(self, user_text, ngen=64, max_trips=2):
        return "served", {
            "mounts": [1], "mount_plan": [0], "mount_fitted": [0],
            "mount_dropped_for_width": [], "ranking_ids": [0], "trip": 0,
        }


class _StepRepository:
    def __init__(self, path):
        self.arena = _StepArena()
        self.path = str(path)
        self.autosave = False
        self.fold_history = []
        self.route_receipt_dir = None

    def _snapshot_state(self):
        return list(self.arena.grafts)

    def _extract_from_new_turns(self, before, context=None):
        return []

    def _librarian(self):
        pass

    def _mark_mutations(self, before):
        pass

    def _page(self):
        return 0

    def flush_now(self):
        pass


def _bind_ledger_methods(repo):
    from core.graft_repository import GraftRepository

    for name in ("configure_route_receipt_ledger", "route_receipt_ledger_dir",
                 "write_route_receipt"):
        setattr(repo, name, getattr(GraftRepository, name).__get__(repo))
    return repo


def test_production_chat_returns_the_receipt_when_no_ledger_configured(tmp_path):
    from core.grm_runtime import (
        ROUTE_RECEIPT_INFO_KEY, ROUTE_RECEIPT_SINK_INFO_KEY, GRMRuntime,
    )

    repo = _bind_ledger_methods(_StepRepository(tmp_path))
    _answer, info = GRMRuntime(repo).chat("q")
    assert info[ROUTE_RECEIPT_SINK_INFO_KEY] == {
        "sink": "info", "reason": "no_ledger_location_configured"}
    record = info[ROUTE_RECEIPT_INFO_KEY]
    assert record["schema"] == ROUTE_RECEIPT_SCHEMA
    assert record["fit"]["final_mounts"] == [0]
    assert record["serving_path"] == "core.grm_runtime.GRMRuntime.chat"
    assert record["provenance"]["request_sha256"] is not None
    assert record["provenance"]["output_sha256"] is not None
    assert record["provenance"]["arena_state_sha256_before_serve"] is not None


def test_production_chat_writes_to_the_repository_ledger_when_configured(
        tmp_path):
    import os

    from core.grm_runtime import (
        ROUTE_RECEIPT_INFO_KEY, ROUTE_RECEIPT_SINK_INFO_KEY, GRMRuntime,
    )

    repo = _bind_ledger_methods(_StepRepository(tmp_path))
    directory = repo.configure_route_receipt_ledger()
    runtime = GRMRuntime(repo)
    _answer, info = runtime.chat("q")
    assert info[ROUTE_RECEIPT_SINK_INFO_KEY]["sink"] == "repository_ledger"
    assert ROUTE_RECEIPT_INFO_KEY not in info
    written = sorted(os.listdir(directory))
    assert len(written) == 1
    on_disk = json.loads(
        open(os.path.join(directory, written[0]), encoding="utf-8").read())
    assert on_disk["schema"] == ROUTE_RECEIPT_SCHEMA
    assert on_disk["turn_id"] == "0"
    # Turn ids advance, so a second serve does not overwrite the first.
    runtime.chat("q2")
    assert len(os.listdir(directory)) == 2
    # And the dump script reads the ledger directory directly.
    receipts = collect_receipts(str(directory))
    assert [r["turn_id"] for r in receipts] == ["0", "1"]


# --------------------------------------------- P2A integration (merge gate)

# The twelve fields LSR-P2A adds to ``info``: eight from
# ``core.grm_admission.fit_info_fields`` and four from the Ruling 2
# abstention branch in ``_probe_ladder_chat``.
P2A_INFO_FIELDS = {
    "fit_planned": [1, 0],
    "fit_seated": [1],
    "fit_dropped_planned": [0],
    "fit_dropped_filler": [],
    "fit_unseatable": [0],
    "fit_shuttle": True,
    "fit_shuttle_trips": [[0]],
    "served_without_plan_head": False,
    "abstained": True,
    "abstain_reason": "identifier_unbound",
    "abstain_identifier_tokens": ["lumen", "seal"],
    "abstain_deposited_graft": 3,
}


def test_all_twelve_p2a_info_fields_land_in_the_route_receipt():
    """P2A's fit-honesty and abstention fields persist with no P2B re-edit.

    Eight arrive by the ``fit_*`` prefix, three by the ``abstain`` prefix, and
    ``served_without_plan_head`` by the explicit key list -- so a P2A-shaped
    ``info`` needs no enumeration here to be recorded.
    """
    info = dict(LUMEN_INFO)
    info.update(P2A_INFO_FIELDS)
    # Private observation hand-off must NOT leak into the pass-through.
    info["_route_observation"] = {"serving_path": "x", "trips": []}
    record = _lumen_receipt(info=info)
    through = record["fit"]["info_pass_through"]
    assert through == P2A_INFO_FIELDS
    assert len(through) == 12
    assert "_route_observation" not in through
    # The named sections still describe the same turn.
    assert record["fit"]["planned"] == [1, 0]
    assert record["fit"]["seated"] == [1]
    assert record["admission"]["policy_branch"] == (
        "declared_synthesis_identified_set")
    # And the whole thing is still canonically stamped.
    assert record["receipt_sha256"] == route_receipt_sha256(record)


def test_the_field_names_match_p2a_fit_info_fields_exactly():
    """Guard against P2A renaming a field out from under the pass-through."""
    from core.grm_admission import fit_info_fields

    emitted = fit_info_fields(
        {
            "fit_planned": [1, 0], "fit_seated": [1],
            "fit_dropped_planned": [0], "fit_dropped_filler": [],
            "fit_unseatable": [0],
        },
        shuttle=True,
        shuttle_trips=[[0]],
        served_without_plan_head=False,
    )
    record = _lumen_receipt(info={**LUMEN_INFO, **emitted})
    through = record["fit"]["info_pass_through"]
    # Every field P2A emits is carried; none is silently dropped.
    assert set(emitted) <= set(through)
    assert set(emitted) == {
        "fit_planned", "fit_seated", "fit_dropped_planned",
        "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
        "fit_shuttle_trips", "served_without_plan_head",
    }
