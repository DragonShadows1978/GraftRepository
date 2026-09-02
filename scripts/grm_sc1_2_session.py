#!/usr/bin/env python3
"""GRM-SC1.2 — pure (CPU) parts of the E2E session-resume driver.

ORDER: ``orders/GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md``.

WHAT THIS MODULE IS.  Everything in SC1.2 that does NOT need a GPU: which
pairs are in play, how the certified session's index state is reconstructed
AT the probe turn, the reproduction comparator, and the 10-pair table
assembly.  The GPU driver (``grm_sc1_2_e2e_recovery_gpu.py``) imports these
and adds only the leased model work, so the reasoning that decides what the
gate MEANS is unit-testable without a card.

THE PROBLEM SC1.2 EXISTS TO SOLVE.  Ten pairs were registered by the race.
SC1/SC1.1's standalone harness reached three: the four ``sup_*`` supersession
sessions are built from a fixture file that ``_install_lived_nodes`` can
replay in any process.  The other seven are ``certified_34_turn`` E2E
fixtures whose lived rows were produced INSIDE a chained 34-turn session by
the campaign worker, and ``scripts/grm_det1_5_workers.py::_context`` refuses
to run anywhere but the frozen run tree.  The flip decision needs ten.

THE PATH TAKEN (registered before the run, path (a) of the order's Mission 1).
Every one of the seven pairs HAS a per-probe-turn lived snapshot inside the
frozen tree -- ``grm.det1_3.model_visible_snapshot.v2``, label ``lived``,
phase ``before_probe_prefill``, finalized.  So the session is not replayed;
it is RESUMED at the probe boundary by the same fork mechanism that produced
the lived rows.  The campaign worker is not bypassed by pointing it at a
copy: it is not called at all, and this driver imports the measurement
functions instead.

THE ONE THING THE SNAPSHOT DOES NOT CARRY, AND HOW IT IS REBUILT.
``restore_prefill_fork`` restores MOUNTED PAYLOAD ONLY.  It explicitly does
not reconstruct the unmounted routing index -- and the demand trip's whole
job is to re-route into that index and fetch the withheld node.  So the
index must be rebuilt, and rebuilt AS OF THE PROBE TURN, not as of whenever
the session happened to be flushed.

  * Node ids in the certified session are deposit order, so the index at a
    probe is exactly nodes ``[0, N)`` for the ``N`` that had been deposited
    when that probe routed.  ``N`` is NOT the number in the fixture id: the
    fixture id carries the zero-based CONVERSATION TURN, and supersede turns
    deposit TWO nodes (the turn plus its extracted fact span), so the two
    counts diverge from turn 7 onward (t13 is conversation turn 14 but node
    15).  ``N`` is therefore DERIVED from the frozen session's own
    ``instrumentation.jsonl``, which records ``nodes_before``/``nodes_after``
    at every deposit -- see :func:`probe_node_counts`.
  * The persisted ``repository/manifest.json`` in each frozen shard session is
    an END-OF-SHARD flush, never a probe-turn state.  Truncating it to
    ``[0, N)`` is necessary but NOT sufficient: ``superseded_by``, ``active``
    and ``retired`` are BACK-EDGES that keep changing as later turns land, so
    an end-of-shard manifest says node 0 is retired-and-inactive at t05, when
    the lived t05 probe demonstrably mounted it.
  * ``supersedes`` is the FORWARD edge and is deposit-final: measured across
    all four frozen shard manifests, the common prefixes agree byte for byte.
    So the probe-turn back-edges are RECOMPUTED as the closure of the
    truncated prefix's own ``supersedes`` edges, and ``active``/``retired``
    follow from it.  Measured against all four frozen manifests at their own
    node counts, that recomputation reproduces the persisted back-edges,
    ``active`` and ``retired`` with ZERO mismatches -- which is why it is
    trusted here rather than assumed.

``no_fold`` is the one flag that cannot be pinned exactly for two of the
seven pairs (see ``NO_FOLD_PINNED``).  It gates ``_foldable``/``consolidate``
-- the librarian -- and nothing else; routing, admission, fit and grounding
never read it, and this driver never runs the librarian.  The ambiguity is
recorded per pair rather than hidden.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: The frozen race run.  READ-ONLY, always: nothing in SC1.2 writes here.
FROZEN_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
CAMPAIGN = FROZEN_RUN / "det1_4" / "campaign"
EVAL_ROWS = CAMPAIGN / "eval" / "mechanistic_rows.jsonl"
SHARDS = CAMPAIGN / "eval" / "mechanistic_shards"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
DET1_REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1_2"
REGISTRATION = ARTIFACT_DIR / "registration.json"
ORDER = ROOT / "orders" / "GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md"

#: SC1.1's frozen 3-pair receipt, merged into the 10-pair table.
SC1_1_G2_SUMMARY = (
    ROOT / "artifacts" / "grm_sc1_1"
    / "sc1_1_g2_summary_c1baaea3f682e7cc.json"
)

#: The seven certified_34_turn pairs, in race row order.
E2E_FIXTURE_IDS = (
    "e2e_t05_orion_pin",
    "e2e_t13_orion_pin",
    "e2e_t19_nova_key",
    "e2e_t22_mira_seal",
    "e2e_t24_terra_port",
    "e2e_t26_ember_code",
    "e2e_t30_atlas_tone",
)

#: The three standalone pairs SC1.1 already measured.
SUP_FIXTURE_IDS = (
    "sup_harbor_restatement",
    "sup_praxis_fresh",
    "sup_solace_fresh",
)

#: Which frozen shard session holds each pair's lived snapshot.
FIXTURE_SHARD = {
    "e2e_t05_orion_pin": "e2e-1",
    "e2e_t13_orion_pin": "e2e-2",
    "e2e_t19_nova_key": "e2e-3",
    "e2e_t22_mira_seal": "e2e-3",
    "e2e_t24_terra_port": "e2e-3",
    "e2e_t26_ember_code": "e2e-4",
    "e2e_t30_atlas_tone": "e2e-4",
}

#: The shard whose persisted repository is the SOURCE of node text/payload.
#: e2e-4 carries all 34 nodes, and its prefix agrees with every earlier
#: shard on text, ntok, kind, sources, tags, rare and ``supersedes`` -- the
#: fields the reconstruction actually reads.  The mutable back-edges are
#: recomputed, never copied, so taking the widest manifest is safe AND is
#: what lets a single source serve every prefix.
SOURCE_SHARD = "e2e-4"

#: ``no_fold`` is a librarian-eligibility flag.  Its ONLY consumers are
#: ``GraftRepository._foldable`` / ``_native_foldable`` / ``consolidate`` --
#: routing, admission, fit and grounding never read it, and this driver never
#: runs the librarian, so it cannot reach any number SC1.2 reports.
#:
#: It is nonetheless reconstructed as faithfully as the frozen evidence
#: allows.  Measured across the four frozen flushes: all-False at 11 and 19
#: nodes, {1,4,6,8} at 28 and 34 nodes.  A probe whose prefix is bracketed by
#: two flushes that AGREE is pinned to that agreed value; a probe whose
#: prefix falls strictly between the 19-node and 28-node flushes cannot be
#: pinned and is named in the receipt as an unpinned-but-inert flag.
#:
#: Keyed by prefix size so it is checked against the DERIVED node count, not
#: against a fixture-id number.
NO_FOLD_FLUSHES = ((11, ()), (19, ()), (28, (1, 4, 6, 8)), (34, (1, 4, 6, 8)))


def no_fold_at_prefix(node_count: int) -> tuple[frozenset[int], bool]:
    """(``no_fold`` node ids at this prefix, whether the value is PINNED).

    Pinned when the nearest flush at-or-below and the nearest flush above
    agree (or when the prefix sits at or beyond the last flush).  Otherwise
    the lower bracket's value is used and ``pinned`` is False, so the receipt
    can say the flag was reconstructed under an ambiguity instead of implying
    it was measured.
    """
    count = int(node_count)
    below: tuple[int, ...] | None = None
    above: tuple[int, ...] | None = None
    for size, ids in NO_FOLD_FLUSHES:
        if size <= count:
            below = ids
        elif above is None:
            above = ids
    if below is None:
        below = ()
    if above is None:
        return frozenset(below), True
    return frozenset(below), tuple(below) == tuple(above)

#: The frozen session whose instrumentation covers all 31 executed turns.
#: The campaign copied each shard's session dir forward, so the LAST shard's
#: ``instrumentation.jsonl`` is the whole chained transcript, not just its own
#: segment (measured: 10 / 17 / 25 / 31 rows across e2e-1..e2e-4, each a
#: prefix of the next).
INSTRUMENTATION_SHARD = "e2e-4"

#: Total nodes in the certified session after its last executed turn.  Used
#: only as the terminal anchor of the BACKWARD walk in
#: :func:`probe_node_counts`; taken from the frozen repository manifest, not
#: from a constant, by the caller that has the manifest in hand.
_FIXTURE_TURN = re.compile(r"^e2e_t(\d+)_")


class SC12Error(RuntimeError):
    """SC1.2 could not be performed as registered."""


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# Pair selection
# --------------------------------------------------------------------------
def race_rows(path: Path | str = EVAL_ROWS) -> dict[str, dict[str, Any]]:
    """The race's own eval rows, keyed ``fixture_id:variant``.  READ-ONLY."""
    return {str(row["row_id"]): dict(row) for row in read_jsonl(path)}


def min_mounted_mass(row: Mapping[str, Any]) -> float | None:
    """The D-NGH score of a race row: min per-token mounted mass."""
    tokens = (row.get("signals") or {}).get("tokens") or ()
    values = [float(t["ngh"]["mounted_mass"]) for t in tokens]
    return min(values) if values else None


def e2e_pairs(
    rows: Mapping[str, Mapping[str, Any]] | None = None,
    node_counts: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    """The seven certified_34_turn pairs, with their lived rows attached.

    ``node_counts`` maps script turn -> nodes present when that turn routed,
    as derived by :func:`probe_node_counts` from the frozen instrumentation.
    It is a required input, not a default: the whole point is that the count
    is EVIDENCE, not arithmetic on a fixture id.

    Fails closed on anything unexpected: a missing arm, a family that is not
    ``certified_34_turn``, a planted-miss row that is not labelled a positive,
    a probe turn with no derived node count, or a plant alias that falls
    outside its own reconstructed prefix.  A quietly dropped pair would
    silently shrink the denominator the flip decision is read off, which is
    exactly the failure this order exists to stop making.
    """
    rows = dict(rows or race_rows())
    if node_counts is None:
        raise SC12Error(
            "e2e_pairs requires node_counts derived from the frozen "
            "instrumentation; the fixture id is NOT a node index")
    out: list[dict[str, Any]] = []
    for fixture_id in E2E_FIXTURE_IDS:
        served = rows.get(f"{fixture_id}:served")
        planted = rows.get(f"{fixture_id}:planted_miss")
        if served is None or planted is None:
            raise SC12Error(f"race rows lack both arms for {fixture_id}")
        for row in (served, planted):
            if str(row.get("source_family")) != "certified_34_turn":
                raise SC12Error(
                    f"{fixture_id} is not a certified_34_turn fixture")
        if int(planted.get("label", -1)) != 1:
            raise SC12Error(f"{fixture_id} planted_miss is not a positive")
        turn = fixture_conversation_turn(fixture_id)
        if turn not in node_counts:
            raise SC12Error(
                f"{fixture_id} maps to script turn {turn}, which the frozen "
                "instrumentation does not record as a probe")
        prefix = int(node_counts[turn])
        aliases = sorted(int(v) for v in planted.get("plant_alias_ids", ()))
        outside = [a for a in aliases if not (0 <= a < prefix)]
        if outside:
            # The withheld node must EXIST in the reconstructed index -- that
            # is the whole premise of a planted miss being recoverable.
            raise SC12Error(
                f"{fixture_id} plant aliases {outside} fall outside its "
                f"reconstructed prefix of {prefix} nodes")
        out.append({
            "fixture_id": fixture_id,
            "source_family": "certified_34_turn",
            "shard": FIXTURE_SHARD[fixture_id],
            "conversation_turn": turn,
            "question": str(served["question"]),
            "expected_values": [
                str(v) for v in served.get("expected_values", ())],
            "rejected_values": [
                *[str(v) for v in served.get("stale_values", ())],
                *[str(v) for v in served.get("wrong_fact_values", ())],
            ],
            "plant_alias_ids": aliases,
            "plant_target_id": planted.get("plant_target_id"),
            "probe_node_index": prefix,
            "no_fold_ids": sorted(no_fold_at_prefix(prefix)[0]),
            "no_fold_pinned": bool(no_fold_at_prefix(prefix)[1]),
            "snapshot_manifest": str(
                (served.get("fork_source_snapshot") or {}).get("path", "")),
            "snapshot_sha256": str(
                (served.get("fork_source_snapshot") or {}).get("sha256", "")),
            "lived": {
                "served_answer": str(served.get("answer", "")),
                "served_correct": bool(served.get("answer_correct")),
                "served_mounted_ids": [
                    int(v) for v in served.get("mounted_ids", ())],
                "served_min_mass": min_mounted_mass(served),
                "served_raw_ranking_ids": [
                    int(v) for v in served.get("raw_ranking_ids", ())],
                "planted_answer": str(planted.get("answer", "")),
                "planted_correct": bool(planted.get("answer_correct")),
                "planted_mounted_ids": [
                    int(v) for v in planted.get("mounted_ids", ())],
                "planted_min_mass": min_mounted_mass(planted),
            },
        })
    if len(out) != len(E2E_FIXTURE_IDS):
        raise SC12Error("E2E pair selection lost a registered pair")
    return out


# --------------------------------------------------------------------------
# Probe-turn index reconstruction
# --------------------------------------------------------------------------
def fixture_conversation_turn(fixture_id: str) -> int:
    """The 1-based script turn an ``e2e_tNN_*`` fixture probes.

    The number in the fixture id is the ZERO-BASED conversation turn, so the
    script turn is one higher.  This is NOT a node index -- see
    :func:`probe_node_counts` for why the two diverge.
    """
    match = _FIXTURE_TURN.match(str(fixture_id))
    if not match:
        raise SC12Error(f"not an E2E fixture id: {fixture_id!r}")
    value = int(match.group(1))
    if value <= 0:
        raise SC12Error(f"E2E fixture id has no positive turn: {fixture_id!r}")
    return value + 1


def probe_node_counts(
    instrumentation: Sequence[Mapping[str, Any]],
    total_nodes: int,
) -> dict[int, int]:
    """How many nodes existed when each probe turn ROUTED, per script turn.

    THE TRAP THIS FUNCTION EXISTS TO AVOID.  It is tempting to read the node
    index straight off the fixture id.  That is wrong from turn 7 onward: a
    ``supersede`` turn deposits TWO nodes (the complete turn, plus the
    extracted fact span), so node index and conversation turn drift apart --
    measured on the frozen session, ``e2e_t13_orion_pin`` is conversation
    turn 14 but the probe deposits node 15, and reconstructing 13 nodes
    instead of 15 would hand the demand trip an index missing the very node
    it is supposed to fetch.

    THE DERIVATION.  The frozen ``instrumentation.jsonl`` records
    ``nodes_before``/``nodes_after`` on every DEPOSIT turn (fact, supersede)
    and nothing on the others (filler, probe) -- and the unrecorded turns are
    exactly the ones that deposit exactly ONE node each.  So the walk runs
    BACKWARD from the known total: an anchored turn takes its recorded
    ``nodes_before``; an unanchored turn is one node below the next turn's
    count.  Backward, not forward: a forward fill would have to guess how many
    nodes an anchored turn added and gets turn 14 wrong by one.

    Cross-checked inside the gate: for every probe, the lived
    ``ranking_ids`` recorded on that same row must lie strictly below the
    derived count.
    """
    rows = list(instrumentation)
    count = len(rows)
    before: list[int | None] = [None] * count
    recorded = [
        (row.get("info") or {}).get("nodes_before") for row in rows]
    for index in range(count - 1, -1, -1):
        if recorded[index] is not None:
            before[index] = int(recorded[index])
            continue
        nxt = before[index + 1] if index + 1 < count else int(total_nodes)
        if nxt is None:
            raise SC12Error(
                f"cannot anchor node count at instrumentation row {index}")
        before[index] = int(nxt) - 1
    out: dict[int, int] = {}
    for index, row in enumerate(rows):
        if str(row.get("kind")) != "probe":
            continue
        value = before[index]
        if value is None or value <= 0:
            raise SC12Error(f"probe at turn {index + 1} has no node count")
        ranking = [
            int(v) for v in (row.get("info") or {}).get("ranking_ids", ()) or ()]
        outside = sorted({i for i in ranking if not (0 <= i < value)})
        if outside:
            # The lived probe routed to a node the derivation says did not
            # exist.  The derivation is wrong; fail closed rather than
            # reconstruct an index that cannot be the lived one.
            raise SC12Error(
                f"probe at turn {index + 1} routed to {outside}, outside a "
                f"derived prefix of {value} nodes")
        out[index + 1] = value
    return out


def derive_lineage(nodes: Sequence[Mapping[str, Any]]
                   ) -> dict[int, dict[str, Any]]:
    """Rebuild ``superseded_by`` / ``active`` / ``retired`` for a node prefix.

    ``supersedes`` is deposit-final: a node's forward edge is written when the
    node lands and never revised, so a truncated prefix carries every forward
    edge that existed at the truncation point and no edge that did not.  The
    back-edge is then the closure of the forward edges INSIDE the prefix, and
    a node is inactive (== retired, measured) exactly when a superseder of it
    is present.

    Pinned against the frozen evidence by :func:`lineage_selfcheck`: run over
    all four frozen shard manifests at their own node counts, this reproduces
    the persisted ``superseded_by``, ``active`` and ``retired`` with zero
    mismatches.
    """
    count = len(nodes)
    back: dict[int, list[int]] = {i: [] for i in range(count)}
    for index, node in enumerate(nodes):
        meta = node.get("metadata") or {}
        for raw in meta.get("supersedes", ()) or ():
            older = int(raw)
            if not (0 <= older < count):
                # A forward edge pointing outside the prefix would mean the
                # prefix rule is wrong.  Fail closed rather than drop it.
                raise SC12Error(
                    f"node {index} supersedes {older}, "
                    f"outside prefix of {count}")
            back[older].append(index)
    return {
        index: {
            "superseded_by": sorted(back[index]),
            "active": not back[index],
            "retired": bool(back[index]),
        }
        for index in range(count)
    }


def truncated_manifest(
    source_manifest: Mapping[str, Any],
    node_count: int,
    *,
    no_fold_at_probe: Mapping[int, bool] | None = None,
) -> dict[str, Any]:
    """A repository manifest describing the index AS OF the probe turn.

    Deposit-time fields (text, ntok, kind, sources, tags, rare, provenance,
    ``supersedes``) are carried verbatim from the frozen source.  The mutable
    back-edges are RECOMPUTED by :func:`derive_lineage`, never copied, because
    an end-of-shard flush records a LATER state than the probe.

    The native checkpoint is dropped: the frozen one indexes all 34 nodes and
    hanging it under a 5-node manifest would be a silent index lie.  Without
    it ``GraftRepository`` rebuilds the native store from the truncated node
    list via ``_sync_native_full``.  ``native_node_id`` is dropped per node
    for the same reason.

    Paging flags (``host_present``/``device_present``/``cold_only``) are reset
    to the freshly-loaded state: they describe where bytes happened to live in
    the lived process's VRAM budget, not what the model could see, and the
    loader re-derives them from the payload files it actually finds.
    """
    nodes = list(source_manifest.get("nodes") or ())
    if node_count > len(nodes):
        raise SC12Error(
            f"probe prefix {node_count} exceeds source manifest {len(nodes)}")
    prefix = [dict(node) for node in nodes[:node_count]]
    lineage = derive_lineage(prefix)
    pinned = dict(no_fold_at_probe or {})
    out_nodes = []
    for index, node in enumerate(prefix):
        meta = dict(node.get("metadata") or {})
        meta["superseded_by"] = list(lineage[index]["superseded_by"])
        meta["active"] = bool(lineage[index]["active"])
        no_fold = bool(pinned.get(index, node.get("no_fold", False)))
        meta["no_fold"] = no_fold
        out_nodes.append({
            "kind": node["kind"],
            "text": node["text"],
            "ntok": int(node["ntok"]),
            "sources": list(node.get("sources") or ()),
            "tags": list(node.get("tags") or ()),
            "rare": list(node.get("rare") or ()),
            "provenance": list(node.get("provenance") or ()),
            "metadata": meta,
            "retired": bool(lineage[index]["retired"]),
            "no_fold": no_fold,
            "durable": True,
            "payload_pending": False,
            "host_present": True,
            "device_present": False,
            "cold_only": False,
        })
    return {
        "dialect": source_manifest["dialect"],
        "dialect_descriptor": source_manifest["dialect_descriptor"],
        "durability_mode": source_manifest["durability_mode"],
        "route_layer": source_manifest["route_layer"],
        "review_buffer": [],
        # The frozen manifest's wal_lsn equals the WAL's own max lsn, so no
        # record replays past it.  The reconstruction ships an EMPTY wal
        # directory as well, so replay can add nothing under any reading.
        "wal_lsn": int(source_manifest.get("wal_lsn", 0)),
        "native_checkpoint": None,
        "nodes": out_nodes,
    }


def lineage_selfcheck(shard_manifests: Mapping[str, Mapping[str, Any]]
                      ) -> dict[str, Any]:
    """Prove :func:`derive_lineage` against the frozen manifests themselves.

    Each frozen shard flush is a real observed lineage state at its own node
    count.  Recomputing from that prefix's forward edges must reproduce it.
    This runs inside the gate, not only in tests, so the receipt carries the
    proof rather than a claim about it.
    """
    rows = []
    for shard, manifest in sorted(shard_manifests.items()):
        nodes = list(manifest.get("nodes") or ())
        derived = derive_lineage(nodes)
        mismatches = []
        for index, node in enumerate(nodes):
            meta = node.get("metadata") or {}
            observed = {
                "superseded_by": sorted(
                    int(v) for v in meta.get("superseded_by", ()) or ()),
                "active": bool(meta.get("active", True)),
                "retired": bool(node.get("retired", False)),
            }
            if observed != derived[index]:
                mismatches.append({
                    "node": index,
                    "observed": observed,
                    "derived": derived[index],
                })
        rows.append({
            "shard": shard,
            "node_count": len(nodes),
            "mismatches": mismatches,
            "match": not mismatches,
        })
    return {
        "schema": "grm.sc1_2.lineage_selfcheck.v1",
        "rule": (
            "superseded_by = closure of the prefix's own deposit-final "
            "`supersedes` forward edges; active = no superseder present; "
            "retired = active is False"),
        "shards": rows,
        "all_match": all(row["match"] for row in rows),
    }


def prefix_agreement(
    source_nodes: Sequence[Mapping[str, Any]],
    shard_nodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deposit-time fields must agree between the source and a shard prefix.

    If they ever disagree, the single-source reconstruction is invalid and the
    gate must stop rather than measure a fabricated index.
    """
    fields = ("text", "ntok", "kind", "sources", "tags", "rare")
    count = len(shard_nodes)
    diffs = []
    for index in range(count):
        left, right = source_nodes[index], shard_nodes[index]
        for field in fields:
            if left.get(field) != right.get(field):
                diffs.append({"node": index, "field": field})
        lsup = [
            int(v) for v in
            (left.get("metadata") or {}).get("supersedes", ()) or ()]
        rsup = [
            int(v) for v in
            (right.get("metadata") or {}).get("supersedes", ()) or ()]
        if lsup != rsup:
            diffs.append({"node": index, "field": "metadata.supersedes"})
    return {"node_count": count, "diffs": diffs, "agree": not diffs}


def ranking_within_prefix(pair: Mapping[str, Any]) -> dict[str, Any]:
    """The lived ranking must lie inside the reconstructed prefix.

    A lived ranking id at or beyond ``N`` would mean the probe saw a node the
    prefix rule says did not exist -- i.e. the rule is wrong.  Fail closed;
    NEVER truncate the ranking to make it fit.
    """
    count = int(pair["probe_node_index"])
    ids = [int(v) for v in pair["lived"]["served_raw_ranking_ids"]]
    outside = sorted({i for i in ids if not (0 <= i < count)})
    return {
        "probe_node_index": count,
        "lived_raw_ranking_ids": ids,
        "outside_prefix": outside,
        "within": not outside,
    }


# --------------------------------------------------------------------------
# Reproduction comparison (Arm 0)
# --------------------------------------------------------------------------
def reproduction_verdict(
    *,
    lived_answer: str,
    fork_answer: str,
    lived_min_mass: float | None,
    fork_min_mass: float | None,
    byte_exact_fork: bool,
) -> dict[str, Any]:
    """Did this arm reproduce its lived row?

    Two independent legs, both reported:

      * TEXT -- the DET1 semantic comparator (``normalize_value_text``), the
        same normalizer the race scored answers with.  Raw byte equality is
        reported ALONGSIDE it, never instead of it: the race's own rule is
        that value comparison is semantics, not glyphs.
      * SIGNAL -- the D-NGH minimum mounted mass.  If the fork is byte-exact
        the minima must be EXACTLY equal (float ==, no tolerance invented
        here).  If it is not byte-exact, the delta is reported with the reason
        instead of being waved through.

    A pair reproduces only when BOTH legs pass.  There is no third leg that
    can rescue a failure, and no tolerance parameter to widen.
    """
    from scripts.grm_det1_common import normalize_value_text

    text_normalized = (
        normalize_value_text(str(lived_answer))
        == normalize_value_text(str(fork_answer)))
    text_bytes = str(lived_answer) == str(fork_answer)
    if lived_min_mass is None or fork_min_mass is None:
        mass_match = False
        delta = None
        mass_reason = "a min mass is missing on one side"
    else:
        delta = float(fork_min_mass) - float(lived_min_mass)
        mass_match = float(fork_min_mass) == float(lived_min_mass)
        mass_reason = (
            "exact float equality (fork is byte-exact)" if byte_exact_fork
            else "fork is NOT byte-exact; delta reported, equality not assumed")
    return {
        "text_normalized_match": bool(text_normalized),
        "text_bytes_match": bool(text_bytes),
        "lived_answer": str(lived_answer),
        "fork_answer": str(fork_answer),
        "lived_min_mass": lived_min_mass,
        "fork_min_mass": fork_min_mass,
        "min_mass_delta": delta,
        "min_mass_exact_match": bool(mass_match),
        "min_mass_comparison_basis": mass_reason,
        "byte_exact_fork": bool(byte_exact_fork),
        "reproduced": bool(text_normalized and mass_match),
    }


def pair_reproduced(arms: Mapping[str, Mapping[str, Any]]) -> bool:
    """A pair reproduces only when BOTH of its arms do."""
    served = (arms.get("served") or {}).get("arm0") or {}
    planted = (arms.get("planted_miss") or {}).get("arm0") or {}
    return bool(served.get("reproduced")) and bool(planted.get("reproduced"))


# --------------------------------------------------------------------------
# Table assembly
# --------------------------------------------------------------------------
def sup_rows_from_sc1_1(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SC1.1's three standalone rows, projected into the 10-pair schema.

    Carried, not re-measured and not re-derived: these are frozen SC1.1
    receipts.  ``planted_measured`` marks the one arm SC1.1 could not measure
    (``sup_solace_fresh``'s planted miss, whose registered plant had no
    mounted seat in its rung) so it is never silently counted as a
    non-detection.
    """
    rows = []
    for row in summary.get("table") or ():
        rows.append({
            "pair": str(row["fixture_id"]),
            "family": "standalone_supersession",
            "instrument": "same_process_fork",
            "source": "SC1.1 G2 (frozen receipt)",
            "arm0_reproduced": None,
            "planted_measured": row.get("planted_fired") is not None,
            "fired": row.get("planted_fired"),
            "token_index": row.get("planted_token_index"),
            "min_mass": row.get("planted_min_mass"),
            "fetched": row.get("planted_fetched"),
            "served_from": row.get("planted_served_from"),
            "value": row.get("planted_served_answer"),
            "recovered": row.get("recovered"),
            "served_control_fired": row.get("served_fired"),
            "served_control_min_mass": row.get("served_min_mass"),
            "served_control_correct": row.get("served_correct"),
        })
    return rows


def e2e_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """SC1.2's seven rows, projected into the 10-pair schema.

    A pair whose Arm 0 did not reproduce carries ``None`` in every Arm 1
    column: it is EXCLUDED, as registered, so it can never be silently
    counted as a detection miss or a recovery failure.
    """
    rows = []
    for pair in pairs:
        arms = pair.get("arms") or {}
        planted = arms.get("planted_miss") or {}
        served = arms.get("served") or {}
        arm1 = planted.get("arm1") or {}
        served_arm1 = served.get("arm1") or {}
        reproduced = bool(pair.get("arm0_reproduced"))
        rows.append({
            "pair": str(pair["fixture_id"]),
            "family": "certified_34_turn",
            "instrument": "cross_process_fork_from_frozen_snapshot",
            "source": "SC1.2 G2/G3",
            "arm0_reproduced": reproduced,
            "planted_measured": bool(reproduced and arm1),
            "fired": arm1.get("demand_fired") if reproduced else None,
            "token_index": (
                arm1.get("demand_token_index") if reproduced else None),
            "min_mass": arm1.get("attempt_min_mass") if reproduced else None,
            "fetched": (
                (arm1.get("demand_trip") or {}).get("demand_fetched")
                if reproduced else None),
            "served_from": arm1.get("demand_served") if reproduced else None,
            "value": arm1.get("served_answer") if reproduced else None,
            "recovered": arm1.get("recovered") if reproduced else None,
            "served_control_fired": (
                served_arm1.get("demand_fired") if reproduced else None),
            "served_control_min_mass": (
                served_arm1.get("attempt_min_mass") if reproduced else None),
            "served_control_correct": (
                (served_arm1.get("served_verdict") or {}).get("correct")
                if reproduced else None),
        })
    return rows


def ten_pair_table(
    e2e_pair_results: Sequence[Mapping[str, Any]],
    sc1_1_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """The order's deliverable: ten registered pairs, one table, honest counts.

    THE FLOOR RULE, applied here and not negotiable: recall / FP / recovery
    are reported over the ACHIEVED count.  A pair that did not reproduce, and
    an arm that was never measurable, are EXCLUDED from both numerator and
    denominator and named in ``excluded_from_arm1``.  The denominator is never
    padded back up to ten to make a rate look like the race's.
    """
    rows = e2e_rows(e2e_pair_results) + sup_rows_from_sc1_1(sc1_1_summary)
    positives = [r for r in rows if r["planted_measured"]]
    negatives = [r for r in rows if r["served_control_fired"] is not None]
    true_positives = sum(1 for r in positives if r["fired"])
    false_fires = sum(1 for r in negatives if r["served_control_fired"])
    recovered = sum(1 for r in positives if r["recovered"])
    recall = (true_positives / len(positives)) if positives else None
    # PASS RULE, carried verbatim from SC1 and registered again before this
    # run: recall >= 0.9 AND false fires <= 1.  Recovery is REPORTED.
    gate_pass = bool(
        recall is not None and recall >= 0.9 and false_fires <= 1)
    excluded = [
        {
            "pair": r["pair"],
            "reason": (
                "Arm 0 did not reproduce -- excluded from Arm 1 as registered"
                if r["arm0_reproduced"] is False
                else "planted-miss arm was not measurable in its rung"),
        }
        for r in rows if not r["planted_measured"]
    ]
    return {
        "schema": "grm.sc1_2.ten_pair_table.v1",
        "registered_pair_count": 10,
        "rows_present": len(rows),
        "measured_positives": len(positives),
        "measured_negatives": len(negatives),
        "true_positives": true_positives,
        "detection_recall": recall,
        "false_fires_on_served_controls": false_fires,
        "recovered": recovered,
        "recovery_rate": (recovered / len(positives)) if positives else None,
        "pass_rule": "recall >= 0.9 AND false_fires <= 1",
        "gate_pass": gate_pass,
        "recovery_is_reported_not_gated": True,
        "excluded_from_arm1": excluded,
        "denominator_note": (
            "Rates are over the ACHIEVED count, never over a padded 10. "
            f"positives={len(positives)}, negatives={len(negatives)}."),
        "instrument_note": (
            "The two halves of this table are NOT the same instrument. The "
            "three sup_* pairs are SAME-PROCESS forks (SC1.1). The seven "
            "e2e_* pairs are CROSS-PROCESS forks from the frozen lived "
            "snapshots, whose routing-index projection restore_prefill_fork "
            "reports as NOT_VERIFIED_CROSS_PROCESS_ZERO_GATE_ONLY. Read the "
            "combined rates with that seam in view."),
        "table": rows,
    }


__all__ = [
    "E2E_FIXTURE_IDS",
    "SUP_FIXTURE_IDS",
    "FIXTURE_SHARD",
    "SOURCE_SHARD",
    "INSTRUMENTATION_SHARD",
    "NO_FOLD_FLUSHES",
    "no_fold_at_prefix",
    "SC12Error",
    "race_rows",
    "min_mounted_mass",
    "e2e_pairs",
    "fixture_conversation_turn",
    "probe_node_counts",
    "derive_lineage",
    "truncated_manifest",
    "lineage_selfcheck",
    "prefix_agreement",
    "ranking_within_prefix",
    "reproduction_verdict",
    "pair_reproduced",
    "sup_rows_from_sc1_1",
    "e2e_rows",
    "ten_pair_table",
]
