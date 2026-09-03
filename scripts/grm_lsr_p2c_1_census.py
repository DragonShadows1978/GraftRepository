#!/usr/bin/env python3
"""GRM-LSR-P2C.1 — pure (CPU) parts of the ten-probe census scoring driver.

ORDER: ``orders/GRM_LSR_P2C_1_E2E_CENSUS_SCORING.md``.
REGISTRATION: ``artifacts/lsr_p2c_1/registration.json`` (written before any
gate ran; every rule below is the rule that file registered).

WHAT THIS MODULE IS.  Everything in P2C.1 that does NOT need a GPU: which ten
probes are in play, where each one's frozen lived snapshot and frozen session
live, how the routing index is reconstructed AS OF each probe turn, the Arm-0
reproduction comparator, and the Arm-0/Arm-1 table assembly.  The GPU driver
(``grm_lsr_p2c_1_gpu.py``) imports these and adds only the leased model work,
so the reasoning that decides what the gates MEAN is unit-testable without a
card.

WHY THIS ORDER EXISTS.  P2C closed the wrong-value class on the supersession
battery, but its G3 -- the ten ``e2e_t*`` probes of the DET1.11 lived-serving
census -- was BLOCKED.  P2C's G3 replayed the certified 34-turn session
through the production driver's INLINE turn path and reproduced 8 of 10; t30
and t33 diverged.  The blocked report diagnosed why:

    The lived census rows were NOT produced by an inline turn path.  Every
    one of them was produced by a COUNTERFACTUAL FORK LADDER -- the arena is
    snapshotted at ``before_probe_prefill``, each rung of the attempt
    schedule is regenerated from that restored base, and ``_select_ladder``
    picks the served rung.  An inline replay is a DIFFERENT SERVING
    MECHANISM, so a divergence there is a statement about the two
    mechanisms, not about the probe.

SC1.2 then built the instrument that matches: fork the frozen per-probe-turn
lived snapshot with ``restore_prefill_fork`` -- the very mechanism that
produced the lived rows -- and rebuild the routing index to the state at that
turn.  On its seven probes it reproduced 7/7 with mass delta exactly 0.0,
including t30, the probe the inline replay lost.  This order carries that
instrument to all ten.

THE MEASURED FACT THAT LICENSES THE CARRY, checked before registration on all
ten probes and re-checked in-gate: the snapshot's ``linked_answer`` records
``probe_selected_attempt = 0`` and ``captured_attempt_ordinal = 0``, and its
``probe_answer`` is BYTE-EQUAL to the census row's ``served_answer``.  So the
lived served answer for every one of the ten was produced at ladder rung 0 --
exactly the rung the frozen snapshot captures.  A probe served from a later
rung would NOT be reachable by this instrument, and the driver refuses to
measure one rather than quietly forking the wrong boundary.

THE THREE PROBES SC1.2 DID NOT REACH, and what is different about them:

  * ``e2e_t09_cypher_bridge`` and ``e2e_t16_lyra_dock`` are CALIBRATION-split
    probes.  Their lived rows come from a different session run than the
    seven eval probes, and that session's repository DISAGREES with the eval
    chain's on deposit-time fields from node 14 onward (a node's text embeds
    the model's own turn output, and the two runs generated different filler
    completions).  So they are reconstructed from the CALIBRATION session's
    own repository.  Reusing the eval superset for them would fabricate an
    index; see :data:`PROBE_SESSIONS`.
  * ``e2e_t33_polaris_mark`` is a RESERVE probe on a turn that is a filler in
    the certified script -- DET1.9 replaced it with the Polaris recall probe.
    Its session therefore runs the full 34 turns (37 nodes) where the eval
    chain stops at 31 turns (34 nodes), and its lived row is a REFUSAL.

THE SIGNAL LEG IS ABSENT ON THOSE THREE, AND THAT IS SAID OUT LOUD.  The
seven eval probes have frozen race rows carrying D-NGH per-token signals, so
Arm 0 compares BOTH the served text and the minimum mounted mass.  The three
new probes' frozen DET1.7 plant-registration observations carry no ``signals``
block at all.  Their Arm-0 verdict is the TEXT leg only, ``mass_leg_available``
is False in their receipt, and an absent mass is never reported as a match.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: The frozen race run.  READ-ONLY, always: nothing in P2C.1 writes here.
FROZEN_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
CAMPAIGN = FROZEN_RUN / "det1_4" / "campaign"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
DET1_REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"

#: The DET1.11 lived-serving census, GENERATION 2.  This is the artifact the
#: order scores against; generation 1 (``..._b6dfcd0fe727ef9c.json``) is the
#: one it supersedes and is never read.
CENSUS = (
    FROZEN_RUN / "det1_11" / "census"
    / "lived_serving_census_98ef71e88dec17a1.json"
)

#: The DET1.7 plant-registration candidate observations: exactly ONE row per
#: census probe (19 rows, 19 census probes, verified 1:1).  This is the single
#: frozen source for each probe's question, expected/stale/wrong-fact values,
#: lived served answer, lived mounts, and its per-probe-turn lived snapshot.
CANDIDATES = (
    CAMPAIGN / "det1_7" / "plant_registration" / "candidate_observations.jsonl")

#: The race's own eval rows -- the ONLY place the D-NGH per-token signals for
#: the seven eval probes live.  The three new probes have no row here.
EVAL_ROWS = CAMPAIGN / "eval" / "mechanistic_rows.jsonl"

ARTIFACT_DIR = ROOT / "artifacts" / "lsr_p2c_1"
REGISTRATION = ARTIFACT_DIR / "registration.json"
ORDER = ROOT / "orders" / "GRM_LSR_P2C_1_E2E_CENSUS_SCORING.md"

#: The ten census probes, in session-turn order.
CENSUS_PROBE_IDS = (
    "e2e_t05_orion_pin",
    "e2e_t09_cypher_bridge",
    "e2e_t13_orion_pin",
    "e2e_t16_lyra_dock",
    "e2e_t19_nova_key",
    "e2e_t22_mira_seal",
    "e2e_t24_terra_port",
    "e2e_t26_ember_code",
    "e2e_t30_atlas_tone",
    "e2e_t33_polaris_mark",
)

#: Which frozen SESSION each probe's index prefix is reconstructed from, and
#: which session's instrumentation derives its N.  Repo-relative, under the
#: frozen run.  The value is the session directory; ``repository/manifest.json``
#: and ``instrumentation.jsonl`` hang off it.
#:
#: THREE FAMILIES, and the reason there are three rather than one:
#:   ``eval``        the seven eval probes -- SC1.2's own chain, whose
#:                   e2e-4 attempt_001 session carries all 31 executed turns
#:                   and all 34 nodes.
#:   ``calibration`` t09 and t16.  A SEPARATE session run; its repository
#:                   diverges from ``eval`` at node 14 (measured), so it is
#:                   its own source and never the eval superset.
#:   ``polaris``     t33.  The DET1.7 e2e-4 attempt_004 session, which runs
#:                   the full 34-turn script including DET1.9's turn-33
#:                   Polaris replacement -- 34 instrumentation rows, 37 nodes.
#:                   The eval chain stops at turn 31 and cannot reach it.
SESSION_FAMILIES = {
    "eval": (
        "det1_4/campaign/eval/mechanistic_shards/e2e-4/attempt_001/session"),
    "calibration": (
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-cal"
        "/attempt_005/session"),
    "polaris": (
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-4"
        "/attempt_004/session"),
}

PROBE_SESSIONS = {
    "e2e_t05_orion_pin": "eval",
    "e2e_t09_cypher_bridge": "calibration",
    "e2e_t13_orion_pin": "eval",
    "e2e_t16_lyra_dock": "calibration",
    "e2e_t19_nova_key": "eval",
    "e2e_t22_mira_seal": "eval",
    "e2e_t24_terra_port": "eval",
    "e2e_t26_ember_code": "eval",
    "e2e_t30_atlas_tone": "eval",
    "e2e_t33_polaris_mark": "polaris",
}

#: Every frozen repository flush in a family, used to (a) prove the lineage
#: recomputation against real observed states and (b) bracket ``no_fold``.
#: Repo-relative session directories.
FAMILY_FLUSHES = {
    "eval": (
        "det1_4/campaign/eval/mechanistic_shards/e2e-1/attempt_001/session",
        "det1_4/campaign/eval/mechanistic_shards/e2e-2/attempt_001/session",
        "det1_4/campaign/eval/mechanistic_shards/e2e-3/attempt_001/session",
        "det1_4/campaign/eval/mechanistic_shards/e2e-4/attempt_001/session",
    ),
    "calibration": (
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-cal"
        "/attempt_001/session",
        "det1_4/campaign/calibration/shards/e2e-cal/attempt_001/session",
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-cal"
        "/attempt_005/session",
    ),
    "polaris": (
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-1"
        "/attempt_004/session",
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-2"
        "/attempt_004/session",
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-3"
        "/attempt_004/session",
        "det1_4/campaign/det1_7/plant_registration/shards/e2e-4"
        "/attempt_004/session",
    ),
}

_FIXTURE_TURN = re.compile(r"^e2e_t(\d+)_")


class P2C1Error(RuntimeError):
    """P2C.1 could not be performed as registered."""


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def session_dir(family: str) -> Path:
    if family not in SESSION_FAMILIES:
        raise P2C1Error(f"unknown session family: {family!r}")
    return FROZEN_RUN / SESSION_FAMILIES[family]


def family_flush_manifests(family: str) -> dict[str, Mapping[str, Any]]:
    """Every frozen repository flush in a family, keyed by its session path.

    Flushes absent from the frozen tree are SKIPPED rather than faked; the
    caller reports how many were found.  A family with no flush at all cannot
    be reconstructed and fails closed here.
    """
    out: dict[str, Mapping[str, Any]] = {}
    for rel in FAMILY_FLUSHES.get(family, ()):
        path = FROZEN_RUN / rel / "repository" / "manifest.json"
        if path.is_file():
            out[rel] = read_json(path)
    if not out:
        raise P2C1Error(f"family {family!r} has no frozen repository flush")
    return out


# --------------------------------------------------------------------------
# no_fold bracketing (per family, from that family's own flushes)
# --------------------------------------------------------------------------
def no_fold_flushes(family: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """``(node_count, no_fold ids)`` for each flush in a family, ascending.

    Read from the frozen manifests, never hard-coded: SC1.2 pinned four
    eval-chain flushes as a literal tuple, and this order needs two more
    families, so the evidence is read instead of transcribed.
    """
    rows = []
    for manifest in family_flush_manifests(family).values():
        nodes = list(manifest.get("nodes") or ())
        ids = tuple(
            index for index, node in enumerate(nodes) if node.get("no_fold"))
        rows.append((len(nodes), ids))
    # Two flushes at the same node count must agree, or the bracket is a lie.
    by_count: dict[int, tuple[int, ...]] = {}
    for count, ids in rows:
        if count in by_count and by_count[count] != ids:
            raise P2C1Error(
                f"family {family!r} has disagreeing no_fold flushes at "
                f"{count} nodes: {by_count[count]} vs {ids}")
        by_count[count] = ids
    return tuple(sorted(by_count.items()))


def no_fold_at_prefix(
    node_count: int,
    flushes: Sequence[tuple[int, Sequence[int]]],
) -> tuple[frozenset[int], bool]:
    """(``no_fold`` ids at this prefix, whether the value is PINNED).

    SC1.2's rule, carried unchanged: pinned when the nearest flush at-or-below
    and the nearest flush above AGREE, or when the prefix sits at or beyond
    the last flush.  Otherwise the lower bracket's value is used and
    ``pinned`` is False, so a receipt can say the flag was reconstructed under
    an ambiguity instead of implying it was measured.

    ``no_fold`` gates ``_foldable`` / ``_native_foldable`` / ``consolidate`` --
    the librarian -- and nothing else.  Routing, admission, fit and grounding
    never read it and this driver never runs the librarian, so an unpinned
    value cannot reach any number this order reports.  It is still
    reconstructed as faithfully as the frozen evidence allows, and the
    ambiguity is named per probe.
    """
    count = int(node_count)
    below: tuple[int, ...] | None = None
    above: tuple[int, ...] | None = None
    for size, ids in flushes:
        if int(size) <= count:
            below = tuple(int(v) for v in ids)
        elif above is None:
            above = tuple(int(v) for v in ids)
    if below is None:
        below = ()
    if above is None:
        return frozenset(below), True
    return frozenset(below), tuple(below) == tuple(above)


# --------------------------------------------------------------------------
# Probe-turn node-count derivation (SC1.2's rule, imported not re-implemented)
# --------------------------------------------------------------------------
def fixture_conversation_turn(probe_id: str) -> int:
    """The 1-based script turn an ``e2e_tNN_*`` probe sits on.

    The number in the id is the ZERO-BASED conversation turn, so the script
    turn is one higher.  This is NOT a node index -- see
    ``grm_sc1_2_session.probe_node_counts`` for why the two diverge.
    """
    match = _FIXTURE_TURN.match(str(probe_id))
    if not match:
        raise P2C1Error(f"not an E2E census probe id: {probe_id!r}")
    value = int(match.group(1))
    if value <= 0:
        raise P2C1Error(f"census probe id has no positive turn: {probe_id!r}")
    return value + 1


def family_probe_node_counts(family: str) -> dict[int, int]:
    """``{script turn: nodes present when that probe routed}`` for a family.

    Delegates to ``grm_sc1_2_session.probe_node_counts`` -- SC1.2's registered
    BACKWARD walk -- applied to THIS family's own instrumentation and its own
    repository node total.  The rule is carried; only the session it is
    applied to changes.  Re-implementing the walk here would let the two
    orders drift apart silently, which is the whole reason it is imported.
    """
    from scripts.grm_sc1_2_session import probe_node_counts

    directory = session_dir(family)
    instrumentation = read_jsonl(directory / "instrumentation.jsonl")
    manifest = read_json(directory / "repository" / "manifest.json")
    return probe_node_counts(instrumentation, len(manifest["nodes"]))


# --------------------------------------------------------------------------
# Probe selection
# --------------------------------------------------------------------------
def census_rows(path: Path | str = CENSUS) -> dict[str, dict[str, Any]]:
    """The census's own rows, keyed by ``probe_id``.  READ-ONLY."""
    payload = read_json(path)
    return {str(row["probe_id"]): dict(row) for row in payload["rows"]}


def candidate_rows(path: Path | str = CANDIDATES) -> dict[str, dict[str, Any]]:
    """The DET1.7 candidate observations, keyed by ``fixture_id``.  READ-ONLY.

    Fails closed on a duplicate: the 1:1 correspondence with the census is the
    premise that lets a single row carry each probe's question, values,
    snapshot and lived answer.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row["fixture_id"])
        if key in out:
            raise P2C1Error(
                f"candidate observations are not 1:1: {key} appears twice")
        out[key] = dict(row)
    return out


def min_mounted_mass(row: Mapping[str, Any]) -> float | None:
    """The D-NGH score of a race row: min per-token mounted mass."""
    tokens = (row.get("signals") or {}).get("tokens") or ()
    values = [float(token["ngh"]["mounted_mass"]) for token in tokens]
    return min(values) if values else None


def eval_signal_rows(path: Path | str = EVAL_ROWS) -> dict[str, dict[str, Any]]:
    """The race's served rows, keyed by fixture id.  The signal source.

    Only the seven eval probes have one.  A probe with no row here has no mass
    leg, which the caller records rather than papering over.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = str(row["row_id"])
        if row_id.endswith(":served"):
            out[row_id[: -len(":served")]] = dict(row)
    return out


def census_probes(
    *,
    census: Mapping[str, Mapping[str, Any]] | None = None,
    candidates: Mapping[str, Mapping[str, Any]] | None = None,
    signals: Mapping[str, Mapping[str, Any]] | None = None,
    node_counts: Mapping[str, Mapping[int, int]] | None = None,
    flushes: Mapping[str, Sequence[tuple[int, Sequence[int]]]] | None = None,
) -> list[dict[str, Any]]:
    """The ten census probes, fully resolved and cross-checked.

    ``node_counts`` maps family -> {script turn: N}, as derived by
    :func:`family_probe_node_counts`.  It is a REQUIRED input, not a default:
    the whole point of the derivation is that N is EVIDENCE, never arithmetic
    on a probe id.

    Fails closed on anything unexpected -- a probe missing from the census or
    the candidates, a served answer that disagrees between the two, an
    expected-value set that disagrees, or a probe with no derived node count.
    A quietly dropped probe would shrink the denominator the PASS RULE is read
    off, which is exactly the failure this order exists not to make.
    """
    census = dict(census or census_rows())
    candidates = dict(candidates or candidate_rows())
    signals = dict(signals if signals is not None else eval_signal_rows())
    if node_counts is None:
        raise P2C1Error(
            "census_probes requires node_counts derived from each family's "
            "frozen instrumentation; the probe id is NOT a node index")
    if flushes is None:
        raise P2C1Error("census_probes requires per-family no_fold flushes")

    out: list[dict[str, Any]] = []
    for probe_id in CENSUS_PROBE_IDS:
        row = census.get(probe_id)
        if row is None:
            raise P2C1Error(f"census has no row for {probe_id}")
        candidate = candidates.get(probe_id)
        if candidate is None:
            raise P2C1Error(
                f"candidate observations have no row for {probe_id}")
        fixture = candidate.get("effective_fixture") or {}

        lived_answer = str(row["served_answer"])
        if str(candidate.get("served_answer", "")) != lived_answer:
            raise P2C1Error(
                f"{probe_id}: census and candidate disagree on the lived "
                "served answer; the 1:1 binding is broken")
        expected = [str(v) for v in fixture.get("expected_values", ())]
        if expected != [str(v) for v in row.get("expected_values", ())]:
            raise P2C1Error(
                f"{probe_id}: census and candidate disagree on expected values")

        family = PROBE_SESSIONS[probe_id]
        turn = fixture_conversation_turn(probe_id)
        counts = node_counts.get(family) or {}
        if turn not in counts:
            raise P2C1Error(
                f"{probe_id} maps to script turn {turn}, which the {family} "
                "session's frozen instrumentation does not record as a probe")
        prefix = int(counts[turn])

        snapshot_rel = str(
            (candidate.get("source_snapshot") or {}).get("path", ""))
        if not snapshot_rel:
            raise P2C1Error(f"{probe_id} has no frozen lived snapshot recorded")

        no_fold_ids, no_fold_pinned = no_fold_at_prefix(
            prefix, flushes.get(family) or ())

        signal_row = signals.get(probe_id)
        lived_min_mass = (
            min_mounted_mass(signal_row) if signal_row is not None else None)

        out.append({
            "probe_id": probe_id,
            "session_family": family,
            "session_dir": SESSION_FAMILIES[family],
            "conversation_turn": turn,
            "probe_node_index": prefix,
            "role": str(row.get("role", "")),
            "split": str(row.get("split", "")),
            "spec": str(row.get("spec", "")),
            "census_attempt": row.get("attempt"),
            "census_class": str(row.get("census_class", "")),
            "census_subclass": str(row.get("census_subclass", "")),
            "separator_artifact_rescued": bool(
                row.get("separator_artifact_rescued")),
            "question": str(fixture["question"]),
            "expected_values": expected,
            "rejected_values": [
                *[str(v) for v in fixture.get("stale_values", ()) or ()],
                *[str(v) for v in fixture.get("wrong_fact_values", ()) or ()],
            ],
            "snapshot_manifest": snapshot_rel,
            "snapshot_sha256": str(
                (candidate.get("source_snapshot") or {}).get("sha256", "")),
            "no_fold_ids": sorted(no_fold_ids),
            "no_fold_pinned": bool(no_fold_pinned),
            # The mass leg exists only where the race recorded D-NGH signals.
            "mass_leg_available": lived_min_mass is not None,
            "lived": {
                "served_answer": lived_answer,
                "served_correct": bool(row["served_answer_correct"]),
                "served_mounted_ids": [
                    int(v) for v in row.get("mounted_ids", ())],
                "served_min_mass": lived_min_mass,
                "census_class": str(row.get("census_class", "")),
            },
        })
    if len(out) != len(CENSUS_PROBE_IDS):
        raise P2C1Error("census probe selection lost a registered probe")
    return out


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------
def snapshot_reachability(probe: Mapping[str, Any]) -> dict[str, Any]:
    """Is this probe's lived row forkable from a frozen snapshot?

    REACHABLE requires every clause below.  A probe that misses any of them is
    UNREACHABLE and is REPORTED as such -- never rebuilt by some other route,
    never patched, never dropped from the table.

      * the snapshot file exists and is the DET1.3 model-visible schema;
      * it is a finalized LIVED capture at ``before_probe_prefill`` (the
        boundary the race itself forked from);
      * its ``linked_answer`` says the lived turn served ladder RUNG 0 and
        its ``probe_answer`` is byte-equal to the census row -- otherwise the
        snapshot is not the boundary that produced the census text;
      * its recorded ``admission.ranking`` lies strictly inside the derived
        prefix ``[0, N)`` -- otherwise the prefix rule is wrong for it.
    """
    path = ROOT / str(probe["snapshot_manifest"])
    detail: dict[str, Any] = {"snapshot_path": str(probe["snapshot_manifest"])}
    if not path.is_file():
        return {
            **detail,
            "reachable": False,
            "reasons": ["frozen lived snapshot is absent from the tree"],
        }
    reasons: list[str] = []
    snapshot = read_json(path)
    provenance = snapshot.get("provenance") or {}
    linked = snapshot.get("linked_answer") or {}
    state = snapshot.get("state") or {}

    if str(snapshot.get("schema")) != "grm.det1_3.model_visible_snapshot.v2":
        reasons.append(f"unexpected snapshot schema {snapshot.get('schema')!r}")
    if str(snapshot.get("label")) != "lived":
        reasons.append("snapshot is not labelled a lived capture")
    if str(provenance.get("arm")) != "lived":
        reasons.append("snapshot provenance arm is not 'lived'")
    if str(snapshot.get("phase")) != "before_probe_prefill":
        reasons.append("snapshot is not at the before_probe_prefill boundary")
    if not snapshot.get("capture_finalized"):
        reasons.append("snapshot capture is not finalized")

    selected = linked.get("probe_selected_attempt")
    captured = linked.get("captured_attempt_ordinal")
    if int(selected if selected is not None else -1) != 0:
        reasons.append(
            f"lived turn served ladder rung {selected}, not rung 0; this "
            "snapshot is not the boundary that produced the census text")
    if int(captured if captured is not None else -1) != 0:
        reasons.append(f"snapshot captured rung {captured}, not rung 0")
    if str(linked.get("probe_answer", "")) != str(
            probe["lived"]["served_answer"]):
        reasons.append(
            "snapshot linked_answer.probe_answer is not byte-equal to the "
            "census served_answer")

    ranking = [int(v) for v in state.get("admission.ranking", ()) or ()]
    prefix = int(probe["probe_node_index"])
    outside = sorted({i for i in ranking if not (0 <= i < prefix)})
    if outside:
        reasons.append(
            f"lived ranking {outside} escapes the derived prefix of {prefix} "
            "nodes; the prefix rule is wrong for this probe")

    detail.update({
        "schema": snapshot.get("schema"),
        "label": snapshot.get("label"),
        "phase": snapshot.get("phase"),
        "capture_finalized": bool(snapshot.get("capture_finalized")),
        "provenance_arm": provenance.get("arm"),
        "provenance_protocol": provenance.get("protocol"),
        "frame_sha256": (snapshot.get("identity") or {}).get("frame_sha256"),
        "probe_selected_attempt": selected,
        "captured_attempt_ordinal": captured,
        "linked_probe_answer_matches_census": (
            str(linked.get("probe_answer", ""))
            == str(probe["lived"]["served_answer"])),
        "lived_admission_ranking": ranking,
        "lived_admission_rank_plan": [
            int(v) for v in state.get("admission.rank_plan", ()) or ()],
        "lived_cur_mounts": [
            int(v) for v in state.get("arena.cur_mounts", ()) or ()],
        "probe_node_index": prefix,
        "ranking_outside_prefix": outside,
        "reachable": not reasons,
        "reasons": reasons,
    })
    return detail


# --------------------------------------------------------------------------
# Arm 0 reproduction comparator
# --------------------------------------------------------------------------
def reproduction_verdict(
    *,
    lived_answer: str,
    fork_answer: str,
    lived_min_mass: float | None,
    fork_min_mass: float | None,
    mass_leg_available: bool,
    byte_exact_fork: bool,
) -> dict[str, Any]:
    """Did this fork reproduce the census's lived served text?

    TEXT is the load-bearing leg and it is the DET1 semantic comparator --
    ``normalize_value_text`` equality, the same normalizer the race scored
    answers with.  Raw byte equality is reported ALONGSIDE it, never instead:
    the race's own rule is that value comparison is semantics, not glyphs.

    MASS is reported when the frozen evidence HAS it.  The seven eval probes
    carry D-NGH signals, so their fork's minimum mounted mass must equal the
    lived minimum exactly (float ==, no tolerance invented here) when the fork
    is byte-exact; when it is not, the delta is reported with the reason.  The
    three probes whose frozen observations carry no signals block have NO mass
    leg: ``mass_leg_available`` is False, the delta is None, and the verdict
    rests on text alone.  That is a NAMED WEAKENING of the instrument on those
    rows -- it is never disguised as a passing mass comparison.

    ``reproduced`` = text matched AND (the mass leg matched, or there is no
    mass leg).  There is no third leg that can rescue a text failure, and no
    tolerance parameter to widen.
    """
    from scripts.grm_det1_common import normalize_value_text

    text_normalized = (
        normalize_value_text(str(lived_answer))
        == normalize_value_text(str(fork_answer)))
    text_bytes = str(lived_answer) == str(fork_answer)

    if not mass_leg_available:
        mass_match = None
        delta = None
        basis = (
            "NO MASS LEG: the frozen lived observation for this probe carries "
            "no D-NGH signals block, so the reproduction verdict rests on the "
            "text leg alone. An absent mass is NOT counted as a match.")
    elif lived_min_mass is None or fork_min_mass is None:
        mass_match = False
        delta = None
        basis = "a min mass is missing on one side despite an available leg"
    else:
        delta = float(fork_min_mass) - float(lived_min_mass)
        mass_match = float(fork_min_mass) == float(lived_min_mass)
        basis = (
            "exact float equality (fork is byte-exact)" if byte_exact_fork
            else "fork is NOT byte-exact; delta reported, equality not assumed")

    reproduced = bool(text_normalized and (mass_match is not False))
    return {
        "text_normalized_match": bool(text_normalized),
        "text_bytes_match": bool(text_bytes),
        "lived_answer": str(lived_answer),
        "fork_answer": str(fork_answer),
        "mass_leg_available": bool(mass_leg_available),
        "lived_min_mass": lived_min_mass,
        "fork_min_mass": fork_min_mass,
        "min_mass_delta": delta,
        "min_mass_exact_match": mass_match,
        "min_mass_comparison_basis": basis,
        "byte_exact_fork": bool(byte_exact_fork),
        "reproduced": reproduced,
    }


# --------------------------------------------------------------------------
# Table assembly
# --------------------------------------------------------------------------
def g2_table(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The order's G2 deliverable: ten probes x N / reachable / reproduced.

    Rates are over the ACHIEVED count.  An UNREACHABLE probe is named and is
    NOT counted as a reproduction failure -- the two are different results and
    conflating them would hide which one happened.
    """
    rows = []
    for result in results:
        arm0 = result.get("arm0") or {}
        reach = result.get("reachability") or {}
        rows.append({
            "probe_id": result["probe_id"],
            "session_family": result["session_family"],
            "conversation_turn": result["conversation_turn"],
            "probe_node_index": result["probe_node_index"],
            "reachable": bool(reach.get("reachable")),
            "unreachable_reasons": list(reach.get("reasons") or ()),
            "lived_answer": result["lived"]["served_answer"],
            "arm0_answer": arm0.get("fork_answer"),
            "reproduced": arm0.get("reproduced"),
            "text_normalized_match": arm0.get("text_normalized_match"),
            "text_bytes_match": arm0.get("text_bytes_match"),
            "mass_leg_available": arm0.get("mass_leg_available"),
            "lived_min_mass": arm0.get("lived_min_mass"),
            "arm0_min_mass": arm0.get("fork_min_mass"),
            "min_mass_delta": arm0.get("min_mass_delta"),
            "index_rank1_match": (
                result.get("index_fidelity") or {}).get("rank1_match"),
            "index_prefix_match": (
                result.get("index_fidelity") or {}).get("prefix_match"),
            "no_fold_pinned": result.get("no_fold_pinned"),
        })
    reachable = [r for r in rows if r["reachable"]]
    reproduced = [r for r in reachable if r["reproduced"]]
    return {
        "schema": "grm.lsr_p2c_1.g2_reproduction.v1",
        "gate": "G2",
        "probe_count": len(rows),
        "reachable_count": len(reachable),
        "unreachable": [
            {"probe_id": r["probe_id"], "reasons": r["unreachable_reasons"]}
            for r in rows if not r["reachable"]
        ],
        "reproduced_count": len(reproduced),
        "prediction": ">= 9/10 reachable and reproduced",
        "gate_pass": bool(len(reproduced) >= 9),
        "not_reproduced": [
            {
                "probe_id": r["probe_id"],
                "lived_answer": r["lived_answer"],
                "arm0_answer": r["arm0_answer"],
                "text_normalized_match": r["text_normalized_match"],
                "min_mass_delta": r["min_mass_delta"],
            }
            for r in reachable if not r["reproduced"]
        ],
        "policy": (
            "NOT-REPRODUCED and UNREACHABLE are RESULTS. A probe that does "
            "not reproduce is EXCLUDED from Arm 1 and is never patched into "
            "reproduction. Rates are over the ACHIEVED count, never padded "
            "back up to ten."),
        "table": rows,
    }


def change_class(lived_correct: bool, arm1_correct: bool | None) -> str:
    """The table's change column.

    ``None`` for ``arm1_correct`` means Arm 1 was not measured for this probe
    (it did not reproduce, or it was unreachable) -- reported as EXCLUDED, not
    silently folded into either direction.
    """
    if arm1_correct is None:
        return "EXCLUDED"
    if lived_correct and not arm1_correct:
        return "REGRESSION"
    if not lived_correct and arm1_correct:
        return "FIX"
    return "SAME"


def g3_table(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The order's G3 deliverable: probe x lived / Arm 0 / Arm 1 / change.

    PASS RULE, registered before the run and not negotiable here: ZERO
    regressions on lived-correct probes.  Everything else -- fixes, t33's
    behaviour, the fit and grounding receipts -- is REPORTED, never gated.

    A probe excluded from Arm 1 is excluded from the regression denominator
    too, and is named in ``excluded_from_arm1``.  It is never counted as a
    regression (it was not measured) and never as a pass (it was not measured
    either).
    """
    rows = []
    for result in results:
        arm0 = result.get("arm0") or {}
        arm1 = result.get("arm1") or {}
        reach = result.get("reachability") or {}
        measured = bool(
            reach.get("reachable") and arm0.get("reproduced") and arm1)
        arm1_correct = (
            bool((arm1.get("served_verdict") or {}).get("correct"))
            if measured else None)
        lived_correct = bool(result["lived"]["served_correct"])
        rows.append({
            "probe_id": result["probe_id"],
            "conversation_turn": result["conversation_turn"],
            "lived_value": result["lived"]["served_answer"],
            "lived_verdict": (
                "CORRECT" if lived_correct
                else result["lived"].get("census_class") or "INCORRECT"),
            "lived_correct": lived_correct,
            "arm0_reproduced": arm0.get("reproduced"),
            "arm1_measured": measured,
            "arm1_value": arm1.get("served_answer") if measured else None,
            "arm1_correct": arm1_correct,
            "arm1_expected_hits": (
                (arm1.get("served_verdict") or {}).get("expected_hits")
                if measured else None),
            "arm1_rejected_hits": (
                (arm1.get("served_verdict") or {}).get("rejected_hits")
                if measured else None),
            "arm1_mounted_ids": arm1.get("mounted_ids") if measured else None,
            "fit_receipt": arm1.get("fit_receipt") if measured else None,
            "grounding_receipt": (
                arm1.get("grounding_receipt") if measured else None),
            "change": change_class(lived_correct, arm1_correct),
        })
    measured_rows = [r for r in rows if r["arm1_measured"]]
    lived_correct_measured = [r for r in measured_rows if r["lived_correct"]]
    regressions = [r for r in rows if r["change"] == "REGRESSION"]
    fixes = [r for r in rows if r["change"] == "FIX"]
    return {
        "schema": "grm.lsr_p2c_1.g3_scoring.v1",
        "gate": "G3",
        "arms": {
            "arm0": "GRM_LSR_FIXES=0, GRM_DEMAND_NGH=0",
            "arm1": "GRM_LSR_FIXES=1 (P2A+P2C+SC1.1), GRM_DEMAND_NGH=0",
        },
        "probe_count": len(rows),
        "arm1_measured_count": len(measured_rows),
        "lived_correct_measured_count": len(lived_correct_measured),
        "regression_count": len(regressions),
        "regressions": [r["probe_id"] for r in regressions],
        "fix_count": len(fixes),
        "fixes": [r["probe_id"] for r in fixes],
        "pass_rule": "0 regressions on lived-correct probes",
        "gate_pass": not regressions,
        "excluded_from_arm1": [
            {
                "probe_id": r["probe_id"],
                "reason": (
                    "Arm 0 did not reproduce -- excluded as registered"
                    if r["arm0_reproduced"] is False
                    else "probe was UNREACHABLE from a frozen lived snapshot"),
            }
            for r in rows if not r["arm1_measured"]
        ],
        "denominator_note": (
            "The regression denominator is the lived-correct probes that were "
            "MEASURED in Arm 1, never a padded ten. "
            f"measured={len(measured_rows)}, "
            f"lived_correct_measured={len(lived_correct_measured)}."),
        "t33_note": (
            "e2e_t33_polaris_mark's lived row is a REFUSAL and no prediction "
            "was registered for it. Its Arm 1 behaviour is REPORTED and does "
            "not enter the PASS RULE, which is scoped to lived-CORRECT "
            "probes."),
        "table": rows,
    }


__all__ = [
    "CENSUS", "CANDIDATES", "EVAL_ROWS", "RUNTIME_FRAME", "DET1_REGISTRATION",
    "ARTIFACT_DIR", "REGISTRATION", "ORDER", "FROZEN_RUN",
    "CENSUS_PROBE_IDS", "SESSION_FAMILIES", "PROBE_SESSIONS", "FAMILY_FLUSHES",
    "P2C1Error",
    "read_json", "read_jsonl", "session_dir", "family_flush_manifests",
    "no_fold_flushes", "no_fold_at_prefix",
    "fixture_conversation_turn", "family_probe_node_counts",
    "census_rows", "candidate_rows", "eval_signal_rows", "min_mounted_mass",
    "census_probes", "snapshot_reachability",
    "reproduction_verdict", "change_class", "g2_table", "g3_table",
]
