"""SUP-WO1 supersession battery, CPU regressions, and guarded GPU harness.

Pytest collects only CPU work from this module: fixture validation, the
frozen L1 score math, and L2 mount-resolution graph cases. The real MiniCPM3
battery (or its Qwen3-4B GQA port) is reachable only through this file's
``--run-gpu`` command-line entry point. A GPU run emits one JSON object per
probe plus a machine-readable threshold-registration input summary; it
intentionally asserts no G0/G1/G2 performance verdict because the plan
requires the lead to register numeric floors after the unpatched G0
measurement.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "supersession_battery"
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, str(REPO_ROOT))

from core.graft_arena import ArenaCache, GQAArenaCache


SCHEMA = "grm_supersession_battery_v1"
RECEIPT_SCHEMA = "grm_supersession_receipt_v1"
DIALECTS = ("mla", "gqa")
DIALECT_DEFAULTS = {
    "mla": {
        "model": "MiniCPM3",
        "arena_type": ArenaCache,
        "route_layer": 44,
        "live_turns": 0,
    },
    "gqa": {
        "model": "Qwen3-4B",
        "arena_type": GQAArenaCache,
        "route_layer": 0,
        "live_turns": 2,
    },
}
SCENARIOS = {
    "short_correction_long_competitor",
    "multi_hop_a_b_c",
    "correction_then_restatement",
    "fresh_fact_controls",
}
NODE_ROLES = {"stale", "correction", "restatement", "competitor", "fresh"}


class FixtureError(ValueError):
    pass


def _dialect_config(args):
    """Resolve the model/arena surface without importing or loading a model."""
    dialect = args.dialect
    if dialect not in DIALECT_DEFAULTS:
        raise ValueError(f"unsupported supersession dialect {dialect!r}")
    config = dict(DIALECT_DEFAULTS[dialect])
    requested_layer = getattr(args, "route_layer", None)
    config["route_layer"] = (
        config["route_layer"] if requested_layer is None
        else int(requested_layer)
    )
    if dialect == "gqa" and config["route_layer"] != 0:
        raise ValueError(
            "GQA supersession battery requires --route-layer 0: Qwen3's "
            "layer-0 |q.k| router is part of the dialect contract")
    config["dialect"] = dialect
    return config


def _receipt_row(record_type, dialect, **fields):
    """Keep every emitted record on the registered receipt schema."""
    if dialect not in DIALECTS:
        raise ValueError(f"unsupported supersession receipt dialect {dialect!r}")
    return {
        "record_type": record_type,
        "schema": RECEIPT_SCHEMA,
        "dialect": dialect,
        **fields,
    }


def _require(condition, message):
    if not condition:
        raise FixtureError(message)


def _string_list(value, where, *, nonempty=False):
    _require(isinstance(value, list), f"{where} must be a list")
    if nonempty:
        _require(value, f"{where} must not be empty")
    for item in value:
        _require(isinstance(item, str) and item.strip(),
                 f"{where} entries must be non-empty strings")
    return value


def _lineage_depth(node_id, nodes_by_id, visiting=None):
    visiting = set() if visiting is None else set(visiting)
    _require(node_id not in visiting,
             f"fixture lineage contains a cycle at {node_id!r}")
    visiting.add(node_id)
    parents = nodes_by_id[node_id]["supersedes"]
    if not parents:
        return 0
    return 1 + max(_lineage_depth(parent, nodes_by_id, visiting)
                   for parent in parents)


def validate_fixture(data, source="<memory>"):
    """Validate one static battery session without importing/loading a model."""
    _require(isinstance(data, dict), f"{source}: top level must be an object")
    _require(data.get("schema") == SCHEMA,
             f"{source}: schema must be {SCHEMA!r}")
    session_id = data.get("session_id")
    _require(isinstance(session_id, str) and session_id.strip(),
             f"{source}: session_id must be a non-empty string")
    scenario = data.get("scenario")
    _require(scenario in SCENARIOS,
             f"{source}: unknown scenario {scenario!r}")
    _require(isinstance(data.get("description"), str)
             and data["description"].strip(),
             f"{source}: description must be non-empty")

    nodes = data.get("nodes")
    _require(isinstance(nodes, list) and len(nodes) >= 2,
             f"{source}: nodes must contain at least two entries")
    nodes_by_id = {}
    for position, node in enumerate(nodes):
        where = f"{source}: nodes[{position}]"
        _require(isinstance(node, dict), f"{where} must be an object")
        node_id = node.get("node_id")
        _require(isinstance(node_id, str) and node_id.strip(),
                 f"{where}.node_id must be a non-empty string")
        _require(node_id not in nodes_by_id,
                 f"{source}: duplicate node_id {node_id!r}")
        _require(node.get("role") in NODE_ROLES,
                 f"{where}.role must be one of {sorted(NODE_ROLES)}")
        _require(isinstance(node.get("text"), str) and node["text"].strip(),
                 f"{where}.text must be non-empty")
        _require(isinstance(node.get("value"), str) and node["value"].strip(),
                 f"{where}.value must be non-empty")
        supersedes = _string_list(node.get("supersedes"),
                                  f"{where}.supersedes")
        _require(len(supersedes) == len(set(supersedes)),
                 f"{where}.supersedes contains duplicates")
        for parent in supersedes:
            _require(parent in nodes_by_id,
                     f"{where}.supersedes target {parent!r} must be an "
                     "earlier node in the scripted session")
        nodes_by_id[node_id] = node

    probes = data.get("probes")
    _require(isinstance(probes, list) and probes,
             f"{source}: probes must be a non-empty list")
    probe_ids = set()
    for position, probe in enumerate(probes):
        where = f"{source}: probes[{position}]"
        _require(isinstance(probe, dict), f"{where} must be an object")
        probe_id = probe.get("probe_id")
        _require(isinstance(probe_id, str) and probe_id.strip(),
                 f"{where}.probe_id must be non-empty")
        _require(probe_id not in probe_ids,
                 f"{source}: duplicate probe_id {probe_id!r}")
        probe_ids.add(probe_id)
        _require(isinstance(probe.get("question"), str)
                 and probe["question"].strip(),
                 f"{where}.question must be non-empty")
        target = probe.get("target_node")
        _require(target in nodes_by_id,
                 f"{where}.target_node {target!r} is unknown")
        correction = probe.get("correction_node")
        _require(correction is None or correction in nodes_by_id,
                 f"{where}.correction_node {correction!r} is unknown")
        stale_nodes = _string_list(probe.get("stale_nodes"),
                                   f"{where}.stale_nodes")
        for stale in stale_nodes:
            _require(stale in nodes_by_id,
                     f"{where}.stale_nodes target {stale!r} is unknown")
        competitor = probe.get("competitor_node")
        _require(competitor in nodes_by_id,
                 f"{where}.competitor_node {competitor!r} is unknown")
        _require(competitor != target,
                 f"{where}: target and competitor must differ")
        expected = _string_list(probe.get("expected_values"),
                                f"{where}.expected_values", nonempty=True)
        stale_values = _string_list(probe.get("stale_values"),
                                    f"{where}.stale_values")
        wrong = _string_list(probe.get("wrong_fact_values"),
                             f"{where}.wrong_fact_values", nonempty=True)
        folded = [value.casefold() for value in expected + stale_values + wrong]
        _require(len(folded) == len(set(folded)),
                 f"{where}: answer-class value lists must be disjoint")

        if scenario == "fresh_fact_controls":
            _require(correction is None and not stale_nodes and not stale_values,
                     f"{where}: fresh controls cannot declare revisions")
        else:
            _require(correction == target,
                     f"{where}: revision probe correction_node must be target_node")
            _require(stale_nodes,
                     f"{where}: revision probe needs at least one stale node")
            ancestors = set()
            stack = [target]
            while stack:
                current = stack.pop()
                for parent in nodes_by_id[current]["supersedes"]:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        stack.append(parent)
            _require(set(stale_nodes) <= ancestors,
                     f"{where}: stale_nodes must be explicit ancestors of target")

    for node_id in nodes_by_id:
        _lineage_depth(node_id, nodes_by_id)

    if scenario == "short_correction_long_competitor":
        probe = probes[0]
        correction_words = len(nodes_by_id[probe["correction_node"]]["text"].split())
        competitor_words = len(nodes_by_id[probe["competitor_node"]]["text"].split())
        _require(competitor_words >= 2 * correction_words,
                 f"{source}: long competitor must have at least twice the "
                 "correction's whitespace-token length")
    elif scenario == "multi_hop_a_b_c":
        _require(max(_lineage_depth(node_id, nodes_by_id)
                     for node_id in nodes_by_id) >= 2,
                 f"{source}: multi-hop fixture must contain A to B to C")
    elif scenario == "correction_then_restatement":
        targets = {probe["target_node"] for probe in probes}
        _require(any(nodes_by_id[node_id]["role"] == "restatement"
                     for node_id in targets),
                 f"{source}: target must be a restatement node")
    else:
        _require(all(not node["supersedes"] for node in nodes),
                 f"{source}: fresh controls cannot contain supersedes edges")

    return {
        "session_id": session_id,
        "scenario": scenario,
        "node_count": len(nodes),
        "probe_count": len(probes),
    }


def load_battery_fixtures(directory=FIXTURE_DIR):
    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    _require(paths, f"no JSON fixtures found under {directory}")
    loaded = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FixtureError(f"{path}: could not load JSON: {exc}") from exc
        summary = validate_fixture(data, str(path))
        loaded.append((path, data, summary))
    return loaded


# ===========================================================================
# CPU fixture-validator tests
# ===========================================================================

def test_battery_fixtures_validate_and_cover_every_g0_scenario():
    loaded = load_battery_fixtures()
    assert len(loaded) == 4
    assert {summary["scenario"] for _, _, summary in loaded} == SCENARIOS
    assert sum(summary["probe_count"] for _, _, summary in loaded) == 5


def test_fixture_validator_rejects_unknown_supersede_target():
    _, data, _ = load_battery_fixtures()[0]
    broken = copy.deepcopy(data)
    broken["nodes"][1]["supersedes"] = ["missing_revision"]
    with pytest.raises(FixtureError, match="must be an earlier node"):
        validate_fixture(broken)


def test_fixture_validator_rejects_missing_probe_diagnostic_role():
    _, data, _ = load_battery_fixtures()[0]
    broken = copy.deepcopy(data)
    broken["probes"][0]["competitor_node"] = "missing_competitor"
    with pytest.raises(FixtureError, match="competitor_node"):
        validate_fixture(broken)


# ===========================================================================
# CPU dialect/receipt tests
# ===========================================================================

def test_dialect_selection_keeps_existing_mla_defaults():
    config = _dialect_config(parse_args([]))

    assert config["dialect"] == "mla"
    assert config["model"] == "MiniCPM3"
    assert config["arena_type"] is ArenaCache
    assert config["route_layer"] == 44
    assert config["live_turns"] == 0


def test_dialect_selection_uses_qwen_gqa_surface():
    config = _dialect_config(parse_args(["--dialect", "gqa"]))

    assert config["dialect"] == "gqa"
    assert config["model"] == "Qwen3-4B"
    assert config["arena_type"] is GQAArenaCache
    assert config["route_layer"] == 0
    assert config["live_turns"] == 2


def test_gqa_dialect_rejects_nonzero_route_layer():
    args = parse_args(["--dialect", "gqa", "--route-layer", "1"])

    with pytest.raises(ValueError, match="requires --route-layer 0"):
        _dialect_config(args)


@pytest.mark.parametrize("dialect", DIALECTS)
def test_receipt_schema_round_trip_preserves_dialect(tmp_path, capsys, dialect):
    path = tmp_path / f"{dialect}.jsonl"
    row = _receipt_row(
        "supersession_battery_run", dialect,
        mode="baseline", debias=False, resolve=False,
    )
    sink = _ReceiptSink(path)
    try:
        sink.emit(row)
    finally:
        sink.close()

    persisted = json.loads(path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert persisted == row
    assert printed == row
    assert persisted["schema"] == RECEIPT_SCHEMA
    assert persisted["dialect"] == dialect


# ===========================================================================
# CPU L1 score tests
# ===========================================================================

def test_length_debias_normalizer_hand_computed_cases():
    # sqrt(log2(K+1)): K=1 -> 1, K=3 -> sqrt(2), K=15 -> 2.
    assert ArenaCache._length_debias_normalizer(1) == 1.0
    assert ArenaCache._length_debias_normalizer(3) == pytest.approx(np.sqrt(2.0))
    assert ArenaCache._length_debias_normalizer(15) == 2.0

    arena = ArenaCache.__new__(ArenaCache)
    arena.length_debias = True
    arena.grafts = [
        {"cent": np.asarray([0.8], dtype=np.float32), "ntok": 1},
        {"cent": np.asarray([1.0], dtype=np.float32), "ntok": 15},
    ]
    out = arena._length_debias_scores({0: 0.8, 1: 1.0}, [0, 1])
    assert out == pytest.approx({0: 0.8, 1: 0.5})


def _bare_route_arena(*, debias):
    arena = GQAArenaCache.__new__(GQAArenaCache)
    arena.length_debias = bool(debias)
    arena.revision_resolution = False
    arena.native_store = None
    arena.last_route_backend = "python"
    arena._cuda_gqa_epoch = 0
    long_key = np.zeros((1, 15, 1), dtype=np.float32)
    long_key[0, 0, 0] = 1.0
    arena.grafts = [
        {"cent": np.asarray([[[0.8]]], dtype=np.float32), "ntok": 1,
         "text": "", "kind": "fact", "retired": False},
        {"cent": long_key, "ntok": 15,
         "text": "", "kind": "fact", "retired": False},
    ]
    arena._probe_key = lambda _text: np.asarray([[[1.0]]], dtype=np.float32)
    return arena


def test_length_debias_flips_long_key_dominance_through_route():
    # Raw latent max: long 1.0 > short 0.8. Debiased: short 0.8 > long 0.5.
    assert _bare_route_arena(debias=False).route("?", exclude=set()) == [1, 0]
    assert _bare_route_arena(debias=True).route("?", exclude=set()) == [0, 1]


def test_length_debias_off_returns_original_score_mapping_unchanged():
    arena = ArenaCache.__new__(ArenaCache)
    arena.length_debias = False
    arena.grafts = []
    base = {4: 0.25, 9: 0.75}
    assert arena._length_debias_scores(base, [4, 9]) is base


# ===========================================================================
# CPU L2 resolution tests (M5 metadata direction, no model)
# ===========================================================================

def _resolution_arena(supersedes, *, enabled=True):
    arena = ArenaCache.__new__(ArenaCache)
    arena.revision_resolution = bool(enabled)
    arena.grafts = [
        {"metadata": {"supersedes": list(parents)}}
        for parents in supersedes
    ]
    return arena


def test_revision_resolution_pairwise_successor_excludes_stale_mount():
    arena = _resolution_arena([[], [0], []])
    assert arena._resolve_revision_mounts([2, 0, 1]) == [2, 1]


def test_revision_resolution_multi_hop_finds_explicit_head():
    # A(0) <- B(1) <- C(2); C excludes both ancestors even if B is absent.
    arena = _resolution_arena([[], [0], [1], []])
    assert arena._resolve_revision_mounts([0, 1, 2, 3]) == [2, 3]
    assert arena._resolve_revision_mounts([0, 2, 3]) == [2, 3]


def test_revision_resolution_no_successor_passthrough_keeps_descent_target():
    arena = _resolution_arena([[], [0], []])
    assert arena._resolve_revision_mounts([0, 2]) == [0, 2]
    assert arena._resolve_revision_mounts([0]) == [0]


def test_revision_resolution_cycle_guard_fails_open():
    arena = _resolution_arena([[1], [0], []])
    assert arena._resolve_revision_mounts([0, 1, 2]) == [0, 1, 2]


def test_revision_resolution_flag_off_preserves_pair():
    arena = _resolution_arena([[], [0]], enabled=False)
    assert arena._resolve_revision_mounts([0, 1]) == [0, 1]
    assert arena._resolve_revision_mounts([1, 0, 1]) == [1, 0, 1]


# ===========================================================================
# Guarded real-model battery
# ===========================================================================

class _ReceiptSink:
    def __init__(self, path=None):
        self._fh = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def emit(self, row):
        line = json.dumps(row, sort_keys=True, ensure_ascii=False)
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()


def _install_fixture_nodes(arena, fixture):
    """Deposit scripted nodes and translate fixture IDs to M5 graph edges."""
    node_to_idx = {}
    for node in fixture["nodes"]:
        idx = int(arena.deposit(node["text"]))
        node_to_idx[node["node_id"]] = idx
        graft = arena.grafts[idx]
        graft["node_id"] = idx
        graft["kind"] = "fact"
        graft["metadata"] = {
            "kind": "fact",
            "active": True,
            "supersedes": [],
            "superseded_by": [],
        }

    for node in fixture["nodes"]:
        idx = node_to_idx[node["node_id"]]
        older = [node_to_idx[parent] for parent in node["supersedes"]]
        arena.grafts[idx]["metadata"]["supersedes"] = older
        for old in older:
            reverse = arena.grafts[old]["metadata"]["superseded_by"]
            if idx not in reverse:
                reverse.append(idx)
    # Metadata changes do not alter route keys, but they do alter L2's graph.
    arena._bump_cuda_gqa_epoch()
    return node_to_idx


def _rank(ranking, idx):
    try:
        return ranking.index(int(idx)) + 1
    except ValueError:
        return None


def _classify_answer(answer, probe):
    folded = str(answer).casefold()
    for value in probe["expected_values"]:
        if value.casefold() in folded:
            return "correct", value
    for value in probe["stale_values"]:
        if value.casefold() in folded:
            return "stale", value
    for value in probe["wrong_fact_values"]:
        if value.casefold() in folded:
            return "wrong-fact", value
    # A hedge/refusal/unregistered value is still outside the correct and
    # stale classes. Preserve that distinction as evidence, not a fourth
    # headline class the work order did not register.
    return "wrong-fact", None


def _mode(args):
    if args.debias and args.resolve:
        return "l1_debias_l2_resolution"
    if args.debias:
        return "l1_debias"
    if args.resolve:
        return "l2_resolution_only"
    return "baseline"


def _load_dialect_model(config):
    """Load only the model selected by the guarded real-battery entry point."""
    from tokenizers import Tokenizer as HFTok

    if config["dialect"] == "mla":
        from core.minicpm3_tc import MiniCPM3_TC, _snap

        tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
        model, model_info = MiniCPM3_TC.from_pretrained()
    elif config["dialect"] == "gqa":
        from core.qwen3_tc import Qwen3_TC, _snap

        tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
        model, model_info = Qwen3_TC.from_pretrained()
    else:  # _dialect_config keeps this fail-closed for normal callers.
        raise ValueError(f"unsupported supersession dialect {config['dialect']!r}")
    return model, tok, model_info


def run_gpu_battery(args, loaded, sink):
    """Run measurement-only probes for the selected dialect; never assert performance."""
    config = _dialect_config(args)
    model, tok, model_info = _load_dialect_model(config)
    mode = _mode(args)
    sink.emit(_receipt_row(
        "supersession_battery_run", config["dialect"],
        mode=mode,
        debias=bool(args.debias),
        resolve=bool(args.resolve),
        model=config["model"],
        model_info=str(model_info),
        performance_status="measurement_only_thresholds_unregistered",
    ))

    receipts = []
    for path, fixture, _summary in loaded:
        # GQA inherits the same early-stop ``step`` path, but its dialect
        # contract fixes the router at layer 0 and retains the gate's two
        # live turns. MLA retains its original layer-44, no-live-turn setup.
        arena = config["arena_type"](
            model,
            encode=lambda text: tok.encode(text).ids,
            decode=lambda ids: tok.decode(ids),
            sink_text="<conversation>\n",
            arena_width=int(args.arena_width),
            route_layer=config["route_layer"],
            topk=int(args.topk),
            live_turns=config["live_turns"],
            cache_deposits=False,
            length_debias=bool(args.debias),
            revision_resolution=bool(args.resolve),
        )
        node_to_idx = _install_fixture_nodes(arena, fixture)
        idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

        for probe in fixture["probes"]:
            ranking = list(arena.route(
                probe["question"], exclude=set(), limit=len(arena.grafts)))
            route_backend = arena.last_route_backend
            answer, info = arena.step(
                probe["question"], ngen=int(args.ngen), deposit=False,
                max_trips=0)
            mounted_indices = [int(node_id) - 1
                               for node_id in info.get("mounts", [])]
            mounted_nodes = [idx_to_node.get(idx, f"graft:{idx}")
                             for idx in mounted_indices]
            target_idx = node_to_idx[probe["target_node"]]
            correction_idx = (node_to_idx[probe["correction_node"]]
                              if probe["correction_node"] is not None
                              else None)
            stale_ranks = {
                node_id: _rank(ranking, node_to_idx[node_id])
                for node_id in probe["stale_nodes"]
            }
            primary_stale_rank = (stale_ranks[probe["stale_nodes"][-1]]
                                  if probe["stale_nodes"] else None)
            competitor_idx = node_to_idx[probe["competitor_node"]]
            classification, matched_value = _classify_answer(answer, probe)
            row = _receipt_row(
                "supersession_probe_receipt", config["dialect"],
                mode=mode,
                session_id=fixture["session_id"],
                scenario=fixture["scenario"],
                fixture=path.name,
                probe_id=probe["probe_id"],
                question=probe["question"],
                route_backend=route_backend,
                ranking_nodes=[idx_to_node.get(idx, f"graft:{idx}")
                               for idx in ranking],
                target_node=probe["target_node"],
                target_rank=_rank(ranking, target_idx),
                correction_node=probe["correction_node"],
                correction_rank=(_rank(ranking, correction_idx)
                                 if correction_idx is not None else None),
                stale_nodes=list(probe["stale_nodes"]),
                stale_rank=primary_stale_rank,
                stale_ranks=stale_ranks,
                competitor_node=probe["competitor_node"],
                competitor_rank=_rank(ranking, competitor_idx),
                mounted_nodes=mounted_nodes,
                mounted_indices=mounted_indices,
                answer_text=answer,
                classification=classification,
                classification_match=matched_value,
                arena_info=info,
            )
            receipts.append(row)
            sink.emit(row)

    counts = {name: sum(row["classification"] == name for row in receipts)
              for name in ("correct", "stale", "wrong-fact")}
    competition = [row for row in receipts
                   if row["correction_rank"] is not None]
    inversions = sum(
        row["competitor_rank"] is not None
        and row["correction_rank"] is not None
        and row["competitor_rank"] < row["correction_rank"]
        for row in competition)
    summary = _receipt_row(
        "supersession_battery_summary", config["dialect"],
        mode=mode,
        probe_count=len(receipts),
        classification_counts=counts,
        threshold_registration_inputs={
            "correction_rank_by_probe": {
                row["probe_id"]: row["correction_rank"] for row in competition
            },
            "stale_rank_by_probe": {
                row["probe_id"]: row["stale_ranks"] for row in competition
            },
            "competitor_rank_by_probe": {
                row["probe_id"]: row["competitor_rank"] for row in receipts
            },
            "competitor_over_correction_inversions": inversions,
            "fresh_control_correct": sum(
                row["scenario"] == "fresh_fact_controls"
                and row["classification"] == "correct"
                for row in receipts),
            "fresh_control_total": sum(
                row["scenario"] == "fresh_fact_controls"
                for row in receipts),
        },
        performance_status="measurement_only_thresholds_unregistered",
    )
    sink.emit(summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-gpu", action="store_true",
        help="Load the selected dialect's model and execute the real battery. "
             "Without this flag, only fixture validation runs.")
    parser.add_argument(
        "--dialect", choices=DIALECTS, default="mla",
        help="Arena/model dialect (default: mla; gqa selects Qwen3-4B).")
    parser.add_argument(
        "--debias", action="store_true",
        help="Enable the default-off L1 log-length-normalized route score.")
    parser.add_argument(
        "--resolve", action="store_true",
        help="Enable the default-off L2 revision-aware mount resolution.")
    parser.add_argument("--arena-width", type=int, default=256)
    parser.add_argument(
        "--route-layer", type=int,
        help="Override the dialect default (MLA: 44; GQA: fixed at 0).")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--ngen", type=int, default=32)
    parser.add_argument(
        "--receipt-jsonl", type=Path,
        help="Optional path that receives the same JSON lines printed to stdout.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = _dialect_config(args)
    sink = _ReceiptSink(args.receipt_jsonl)
    try:
        loaded = load_battery_fixtures()
        sink.emit(_receipt_row(
            "supersession_fixture_validation", config["dialect"],
            fixture_count=len(loaded),
            probe_count=sum(summary["probe_count"]
                            for _, _, summary in loaded),
            scenarios=sorted(summary["scenario"]
                             for _, _, summary in loaded),
            status="ok",
        ))
        if not args.run_gpu:
            sink.emit(_receipt_row(
                "supersession_battery_skip", config["dialect"],
                status="gpu_gate_not_requested",
                gpu_work_done=False,
            ))
            return 0
        run_gpu_battery(args, loaded, sink)
        return 0
    finally:
        sink.close()


if __name__ == "__main__":
    raise SystemExit(main())
