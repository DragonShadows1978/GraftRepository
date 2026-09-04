#!/usr/bin/env python3
"""GRM-RS3 — measure the capture pin and the seat-near-live lever.

ORDER: ``orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md`` (Part 3).
REGISTRATION: ``artifacts/grm_rs3/registration.json`` (written FIRST, by
``scripts/grm_rs3_registration.py``; this module refuses to run without it and
never rewrites it).

WHAT IS DIFFERENT FROM RS2, and it is the whole point.  RS2 measured both
levers through SEAMS THAT LIVED IN THE PROBE SCRIPT: ``_capture_and_mount``
poked ``self_attn.live_shift`` around a harvest, and ``_positioned_injection``
built its own injection block.  RS3 moved both into PRODUCTION, behind flags
that default OFF, and this module drives the PRODUCTION path with the flags
set — it implements no seam of its own.  Concretely:

  * the capture pin is ``ArenaCache.deposit(text, capture_pin=...)``, so a
    re-capture here is production's own deposit with a registered geometry;
  * the seating lever lives inside ``ArenaCache._attempt``'s bootstrap branch,
    so a seated arm is a PRODUCTION serve — ``_attempt`` itself, not an
    inlined copy of its decode loop.  RS2's B3a could not use ``_attempt``
    (its bootstrap would have overwritten the arm's positioned injection); RS3
    can, because the seating IS the bootstrap now.

That difference is why C2/C3 can be judged on GENERATION and RS2's B3a could
not: B3a's inlined decode ran over a block whose rows had been SHEARED (see the
constant-delta law in the registration), and it emitted ``<|start|>`` garbage.

THE INSTRUMENT is RS2's ``LayerTypeMassObserver``, IMPORTED unchanged, so the
full-layer numbers stay directly comparable to RS2's and to a production demand
receipt, and the sliding-layer partition is the one RS2 registered.

THE BUILD is RS1's, imported: ``_load_lived_repo`` then
``grm_det1_3_gpu._install_lived_nodes``, and the fixture's WHOLE probe plan is
replayed in the lived order for the reason RS1 gave — serving a target probe
alone is a different computation and would not be a reproduction.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via ``grm_cmc1_gpu_arms.gpu_lease``; the
    operator has ABSOLUTE right of way and this module never signals, kills or
    interrupts any process — it waits on the flock or reports blocked;
  * ONE arm over ONE fixture per process, one lease per process, under the
    house cap; the caller inserts the inter-process gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_frame import CAPTURE_PIN_ENV, SEAT_NEAR_LIVE_ENV  # noqa: E402
from core.grm_frame import ENV_NAME as PERSISTENT_BOAT_ENV  # noqa: E402
from core.grm_frame import env_persistent_boat  # noqa: E402
from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_rs1_read_strength_gpu import (  # noqa: E402
    _feed_live, _load_lived_repo, _probe_plan, answer_verdict)
from scripts.grm_rs2_mount_read_gpu import (  # noqa: E402
    FULL, SLIDING, LayerTypeMassObserver, _clear_boat, _layer_inventory,
    _restore_native_ids, _snapshot_native_ids, payload_digest,
)
from scripts.grm_rs3_registration import RS3Error  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs3"
REGISTRATION = ARTIFACT_DIR / "registration.json"
SCHEMA_PREFIX = "grm.rs3"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 560
LOCK_WAIT_SECONDS = 7200

#: Every arm this module serves, in registration order.
ARMS = ("C0", "C1m", "C1l", "C2", "C3m", "C3l", "C5")

#: The arms whose mount set is a graft this module RE-CAPTURED under a pin.
CAPTURE_ARMS = ("C1m", "C1l", "C3m", "C3l")


# ======================================================================
# Registration reading (pure — CPU tested)
# ======================================================================
def read_registration(path: Path = REGISTRATION) -> dict[str, Any]:
    if not Path(path).is_file():
        raise RS3Error(
            f"registration missing: {path}. Write it FIRST with "
            "scripts/grm_rs3_registration.py; no gate may run without it.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_probes(registration: Mapping[str, Any]) -> dict[str, Any]:
    return dict(registration["probes_REGISTERED_BEFORE_ANY_GATE"])


def probes_for_session(registration: Mapping[str, Any],
                       session_id: str) -> list[str]:
    return sorted(
        probe_id
        for probe_id, probe in registered_probes(registration).items()
        if str(probe["session_id"]) == str(session_id))


def sessions_in_play(registration: Mapping[str, Any]) -> list[str]:
    return sorted({
        str(probe["session_id"])
        for probe in registered_probes(registration).values()})


def arm_flags(registration: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """The two lever values this arm runs at, READ OFF THE REGISTRATION.

    Never typed here: an arm's flags are what the registration says they are,
    so a table row cannot claim a geometry the run did not use.
    """
    arms = registration["arms_REGISTERED_BEFORE_ANY_GATE"]
    if arm not in arms:
        raise RS3Error(f"unknown arm {arm!r}; registered arms are {ARMS}")
    entry = arms[arm]
    return {
        "capture_pin": str(entry["capture_pin"]),
        "seat_near_live": bool(entry["seat_near_live"]),
        "label": str(entry["label"]),
    }


def solace_variant_applies(registration: Mapping[str, Any],
                           probe_id: str) -> bool:
    variant = registration.get(
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE") or {}
    return str(variant.get("probe_id", "")) == str(probe_id)


def arm_plan(registration: Mapping[str, Any],
             session_id: str) -> list[dict[str, Any]]:
    """One work item per (probe, arm, variant) for this fixture.

    The solace probe additionally gets the ``solace_fact`` variant of EVERY
    arm, for the reason RS2 recorded: RS1's A0 mounted a FIT-TIME SPLIT CHILD
    of the long ``sable_competitor`` on that probe, not the solace fact, so a
    read-strength arm run only on the registered mount would be confounded by
    the wrong-graft finding.
    """
    probes = registered_probes(registration)
    variant_spec = registration.get(
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE") or {}
    items: list[dict[str, Any]] = []
    for probe_id in probes_for_session(registration, session_id):
        probe = probes[probe_id]
        for arm in ARMS:
            items.append({
                "probe_id": probe_id,
                "session_id": str(probe["session_id"]),
                "arm": arm,
                "variant": "registered",
                "mount_ids": [int(v) for v in probe["a0_mounted_ids"]],
                "live_node_ids": [str(v) for v in probe["live_node_ids"]],
            })
            if solace_variant_applies(registration, probe_id):
                items.append({
                    "probe_id": probe_id,
                    "session_id": str(probe["session_id"]),
                    "arm": arm,
                    "variant": "solace_fact",
                    "mount_ids": [int(variant_spec["graft_id"])],
                    "live_node_ids": [str(v)
                                      for v in probe["live_node_ids"]],
                })
    return items


#: The registration carries RS2's field names (it inherits RS2's probe block
#: verbatim). ``capture_text_B0_B1`` is the STANDALONE deposit text — the text
#: the lived installer captured from — which is exactly the text an RS3
#: re-capture must use, since RS3 varies only the capture GEOMETRY.  RS2's B2
#: scaffold variant is a different TEXT and is deliberately not read here: no
#: RS3 arm varies the text.
CAPTURE_TEXT_FIELD = "capture_text_B0_B1"


def capture_text_for(registration: Mapping[str, Any], probe_id: str,
                     graft_id: int) -> dict[str, Any] | None:
    """The registered capture text for a fixture node, or ``None`` for a split
    child (which has only its own stored text to re-capture from)."""
    probe = registered_probes(registration)[probe_id]
    for node in probe.get("capture_nodes", ()):
        if int(node.get("graft_id", -1)) != int(graft_id):
            continue
        if not node.get("is_fixture_node", False):
            # A FIT-TIME SPLIT CHILD. The registration records that it is not
            # a fixture node and carries no fixture text for it, deliberately:
            # the only text such a child has is its own stored text, which is
            # what a re-capture of it must use. Fall through to that.
            return None
        if CAPTURE_TEXT_FIELD not in node:
            raise RS3Error(
                f"{probe_id}: fixture capture node {graft_id} carries no "
                f"{CAPTURE_TEXT_FIELD!r}; the registration's probe block did "
                "not come from RS2's schema as expected")
        return {"text": str(node[CAPTURE_TEXT_FIELD]),
                "source": f"registration.capture_nodes.{CAPTURE_TEXT_FIELD}",
                "node_id": node.get("node_id"),
                "registered_sha256": node.get(
                    f"{CAPTURE_TEXT_FIELD}_sha256")}
    variant = registration.get(
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE") or {}
    if (str(variant.get("probe_id", "")) == str(probe_id)
            and int(variant.get("graft_id", -1)) == int(graft_id)):
        return {"text": str(variant["capture_text"]),
                "source": "registration.solace_fact_variant.capture_text",
                "node_id": variant.get("node_id"),
                "registered_sha256": variant.get("capture_text_sha256")}
    return None


# ======================================================================
# The re-capture, THROUGH PRODUCTION's deposit
# ======================================================================
def _recapture_under_pin(repo, arena, registration: Mapping[str, Any],
                         probe_id: str, arm: str, capture_pin: str,
                         mount_ids: Sequence[int]) -> dict[str, Any]:
    """Re-deposit the arm's node(s) with the pin ON, and return the fresh ids.

    THE DEPOSIT PATH IS PRODUCTION'S.  ``arena.deposit(text,
    capture_pin=...)`` — the same call ``feed``'s ephemeral branch makes for
    every lived node, with the one registered argument set.  RS2 had to reach
    into ``self_attn.live_shift`` from the probe script to do this; RS3 does
    not, and that is the seam this order built.

    THE NATIVE SYNC IS NOT OPTIONAL and it is production's own call.  A graft
    appended straight onto ``arena.grafts`` carries no ``native_node_id``, so
    the first ``_attempt`` over it dies in ``_commit_native_mount``.  The lived
    installer ends by calling ``repo._native_sync_node(index)`` for every node;
    this does the same, through the same call.  (Measured on RS2's first B1
    run, with a log — carried forward rather than rediscovered.)
    """
    fresh: list[int] = []
    detail: list[dict[str, Any]] = []
    for graft_id in [int(v) for v in mount_ids]:
        registered = capture_text_for(registration, probe_id, graft_id)
        if registered is not None:
            text = str(registered["text"])
            source = str(registered["source"])
            node_id = registered.get("node_id")
            # The registered digest is CHECKED, not merely carried: an arm
            # that re-captured a different text than the one registered would
            # be measuring a second variable.
            want = registered.get("registered_sha256")
            got = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if want and str(want) != got:
                raise RS3Error(
                    f"{probe_id}/{arm}: registered capture text for graft "
                    f"{graft_id} hashes to {got} but the registration says "
                    f"{want}")
        else:
            text = str(arena.grafts[graft_id].get("text", ""))
            source = "graft_own_text (fit-time split child)"
            node_id = None
            if not text:
                raise RS3Error(
                    f"{probe_id}/{arm}: graft {graft_id} carries no text to "
                    "re-capture from")
        installed = payload_digest(arena.grafts[graft_id])
        new_id = int(arena.deposit(text, capture_pin=capture_pin))
        recaptured = payload_digest(arena.grafts[new_id])
        native_id = repo._native_sync_node(new_id)
        arena._bump_cuda_gqa_epoch()
        fresh.append(new_id)
        stored = arena.grafts[new_id]
        detail.append({
            "installed_graft_id": int(graft_id),
            "fresh_graft_id": new_id,
            "fresh_native_node_id": (
                None if native_id is None else int(native_id)),
            "deposit_path": "ArenaCache.deposit(text, capture_pin=...)",
            "deposit_path_is_production": True,
            "native_sync_call": "GraftRepository._native_sync_node",
            "capture_text_source": source,
            "node_id": node_id,
            "capture_text_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "capture_text": text,
            "installed_payload": installed,
            "recaptured_payload": recaptured,
            "payload_identical": bool(
                installed.get("available") and recaptured.get("available")
                and installed["sha256"] == recaptured["sha256"]),
            "ntok_identical": bool(
                installed.get("ntok") == recaptured.get("ntok")),
            # The receipt the pin stamped onto the graft itself.
            "capture_receipt": {
                key: stored.get(key)
                for key in ("capture_pin", "capture_shift",
                            "capture_shift_observed",
                            "capture_shift_derived_from",
                            "n_sink", "arena_width", "live_shift")
            },
        })
    return {
        "arm": arm,
        "capture_pin": str(capture_pin),
        "fresh_graft_ids": fresh,
        "per_node": detail,
        "all_payloads_identical": bool(
            detail and all(row["payload_identical"] for row in detail)),
    }


# ======================================================================
# The serves
# ======================================================================
def _serve_ladder(repo, e2e, question: str, flags: Mapping[str, Any],
                  ) -> tuple[str, dict[str, Any], list[int], dict[str, Any]]:
    """C0: the PRODUCTION probe path, observed with RS2's instrument."""
    from core import grm_demand

    arena = repo.arena
    with LayerTypeMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
    ) as observer:
        answer, info = e2e._probe_ladder_chat(
            repo, question,
            topk=int(flags["topk"]),
            ngen=int(flags["ngen"]),
            max_trips=int(flags["max_trips"]),
            defer_memory=True,
        )
    return (str(answer), dict(info or {}),
            [int(v) for v in arena.cur_mounts], observer.partition())


def _serve_direct(repo, e2e, question: str, flags: Mapping[str, Any],
                  picks: Sequence[int],
                  ) -> tuple[str, dict[str, Any], list[int], dict[str, Any]]:
    """Every non-C0 arm: ONE ``_attempt`` over an EXPLICIT mount set.

    THIS IS PRODUCTION'S ``_attempt``, including for the seated arms.  RS2's
    positioned arm could not use it — its bootstrap branch would have
    overwritten the probe script's positioned injection — so B3a inlined a copy
    of the decode loop.  RS3's seating lever lives INSIDE that bootstrap
    branch, so C2/C3 run the real thing: the real prompt forward, the real
    early-stop rule, the real final commit forward, the real grounding.
    """
    from core import grm_demand

    arena = repo.arena
    requested = [int(v) for v in picks]
    fitted = [int(v) for v in e2e._budget_fit_mounts(arena, requested)]
    dropped = [v for v in requested if v not in set(fitted)]
    with LayerTypeMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
    ) as observer:
        answer, info = arena._attempt(
            question, fitted, int(flags["ngen"]), False,
            arena.stop_sequences or (), defer_memory=True)
    mounts = [int(v) for v in arena.cur_mounts]
    grounded, _c = arena._grounding_attribution(str(answer), mounts, question)
    out = dict(info or {})
    out["rs3_grounded"] = bool(grounded)
    out["rs3_serving_path"] = (
        "ArenaCache._attempt (routing disabled) — PRODUCTION, including the "
        "bootstrap seating branch the seat lever lives in")
    out["rs3_fit_requested"] = requested
    out["rs3_fit_seated"] = fitted
    out["rs3_fit_dropped"] = dropped
    out["rs3_arena_width"] = int(arena.width)
    out["rs3_cur_mount_n_at_serve"] = int(arena.cur_mount_n)
    # The seating receipt _attempt's bootstrap branch stamped for THIS serve.
    out["rs3_seating"] = getattr(arena, "_rs3_last_seating", None)
    return str(answer), out, mounts, observer.partition()


def _prelude(repo, e2e, question: str, flags: Mapping[str, Any], arena,
             mount_ids: Sequence[int]) -> dict[str, Any]:
    """C0's own ladder, run first so the repository is the one C0 had here.

    RS1 measured why this is not optional: two of the registered mount ids are
    FIT-TIME SPLIT CHILDREN that do not exist until a fit has run, and the
    split is PERSISTED, so requesting them on a freshly installed repository
    raises out of ``_budget_fit_mounts``.
    """
    answer, info, mounts, _mass = _serve_ladder(repo, e2e, question, flags)
    return {
        "why": (
            "C0's own ladder was run first so this arm's repository is the "
            "one C0 had at this probe — including any PERSISTED fit-time "
            "split whose child the registered mount ids name"),
        "served_answer": str(answer),
        "mounted_ids": [int(v) for v in mounts],
        "fit_split_parent": info.get("fit_split_parent"),
        "fit_split_children": info.get("fit_split_children"),
        "grafts_after_prelude": len(arena.grafts),
        "arm_mount_ids": [int(v) for v in mount_ids],
    }


# ======================================================================
# Row and table assembly (pure — CPU tested)
# ======================================================================
def _row(probe: Mapping[str, Any], probe_id: str, session_id: str, arm: str,
         variant: str, registered: bool, question: str, answer: str,
         mounts: Sequence[int], picks_requested: Sequence[int] | None,
         feed_receipt: Mapping[str, Any] | None, mass: Mapping[str, Any],
         info: Mapping[str, Any], mount_n: int, started: int,
         levers: Mapping[str, Any]) -> dict[str, Any]:
    verdict = answer_verdict(
        answer,
        expected_values=probe["expected_values"],
        rejected_values=probe["rejected_values"])
    return {
        "probe_id": probe_id,
        "session_id": session_id,
        "arm": arm,
        "variant": variant,
        "registered_probe": bool(registered),
        "levers": dict(levers),
        "question": question,
        "expected_values": list(probe["expected_values"]),
        "rejected_values": list(probe["rejected_values"]),
        "served_answer": str(answer),
        "correct": bool(verdict["correct"]),
        "verdict": verdict,
        "mounted_ids": [int(v) for v in mounts],
        "picks_requested": (
            None if picks_requested is None
            else [int(v) for v in picks_requested]),
        "live_feed": (None if feed_receipt is None else dict(feed_receipt)),
        "mass": dict(mass),
        "info": {
            key: value for key, value in dict(info).items()
            if not str(key).startswith("_")
        },
        "eb1_lived_answer": str(probe["lived_answer"]),
        "arena_cur_mount_n": int(mount_n),
        "elapsed_ns": int(time.time_ns() - started),
    }


def assemble_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The G3 table: one row per (probe, variant, arm), in that order.

    The columns are exactly the ones the order names: served, correct, mounted
    mass full, mounted mass sliding, live mass, sink mass, top layer, graft
    digest, seat offset.  Pure shaping over already-measured rows.

    THE MASS COLUMNS ARE READ THE WAY RS2 READ THEM, key for key, off the same
    ``LayerTypeMassObserver.partition()`` structure.  That is what makes a C0
    cell and a B0 cell the same number rather than two computations that
    happen to agree — which is exactly what gate G2 has to check.
    """
    order = {arm: index for index, arm in enumerate(ARMS)}
    out: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda r: (str(r["probe_id"]), str(r.get("variant", "registered")),
                       order.get(str(r["arm"]), len(order))),
    ):
        mass = dict(row.get("mass") or {})
        by_type = dict(mass.get("by_layer_type") or {})
        full = dict(by_type.get(FULL) or {})
        sliding = dict(by_type.get(SLIDING) or {})
        full_mean = dict(full.get("mean_over_answer_positions") or {})
        sliding_mean = dict(sliding.get("mean_over_answer_positions") or {})
        top_full = dict(full.get("top_contributing_layer") or {})
        top_sliding = dict(sliding.get("top_contributing_layer") or {})
        info = row.get("info") or {}
        capture = info.get("rs3_capture") or {}
        seating = info.get("rs3_seating") or {}
        digests = [node.get("recaptured_payload", {}).get("sha256")
                   for node in capture.get("per_node", ())]
        out.append({
            "probe_id": str(row["probe_id"]),
            "variant": str(row.get("variant", "registered")),
            "arm": str(row["arm"]),
            "levers": dict(row.get("levers") or {}),
            "served": str(row["served_answer"]),
            "correct": bool(row["correct"]),
            "mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
            "mounted_mass_full_layers": full_mean.get("mounted_mass"),
            "mounted_mass_sliding_layers": sliding_mean.get("mounted_mass"),
            "live_mass_full_layers": full_mean.get("live_mass"),
            "live_mass_sliding_layers": sliding_mean.get("live_mass"),
            "sink_mass_full_layers": full_mean.get("physical_sink_mass"),
            "sink_mass_sliding_layers": sliding_mean.get(
                "physical_sink_mass"),
            "learned_sink_mass_full_layers": full_mean.get(
                "learned_sink_mass"),
            "top_layer_full": top_full.get("layer_ordinal"),
            "top_layer_full_mounted_mass": top_full.get("mounted_mass"),
            "top_layer_sliding": top_sliding.get("layer_ordinal"),
            "top_layer_sliding_mounted_mass": top_sliding.get("mounted_mass"),
            "sliding_window_reaches_mount": (
                dict(mass.get("sliding_window_reach") or {})
                .get("always_reaches_mount")),
            "answer_positions": mass.get("answer_positions"),
            # RS3's own columns, on top of RS2's.
            "graft_digest": [d for d in digests if d],
            "capture_pin_applied": capture.get("capture_pin"),
            "seat_offset_plan_head": seating.get("seat_offset_plan_head"),
            "seat_order": seating.get("seat_order"),
            "mount_pos0": seating.get("mount_pos0"),
            "delta_positions": seating.get("delta_positions"),
            "seat_near_live_applied": seating.get("seat_near_live"),
        })
    return out


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    blob = canonical_json_bytes(payload)
    digest = hashlib.sha256(blob).hexdigest()
    path = ARTIFACT_DIR / f"{stem}_{digest[:16]}.json"
    path.write_bytes(blob)
    return path


# ======================================================================
# The arm runner
# ======================================================================
def run_arm(session_id: str, arm: str) -> dict[str, Any]:
    """One arm over ONE fixture: build, then serve the fixture's probe plan."""
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual)

    if arm not in ARMS:
        raise RS3Error(f"unknown arm {arm!r}; registered arms are {ARMS}")
    registration = read_registration()
    targets = set(probes_for_session(registration, session_id))
    if not targets:
        raise RS3Error(f"no registered probe belongs to fixture {session_id}")
    levers = arm_flags(registration, arm)
    plan_items = [item for item in arm_plan(registration, session_id)
                  if item["arm"] == arm]

    frame = json.loads(RUNTIME_FRAME.read_text(encoding="utf-8"))
    flags = frame["resolved_flags"]
    fixture = json.loads(
        (SUP_FIXTURES / f"{session_id}.json").read_text(encoding="utf-8"))
    plan = _probe_plan(session_id)

    # The lived switches, pinned for this process.  Fixes ON (production),
    # demand OFF (the order: "fixes ON, demand OFF"), spec frame (the escape
    # left UNSET so the fail-closed default selects the ephemeral boat).
    os.environ["GRM_LSR_FIXES"] = "1"
    os.environ["GRM_DEMAND_NGH"] = "0"
    os.environ.pop(PERSISTENT_BOAT_ENV, None)
    # THE TWO NEW LEVERS ARE PASSED EXPLICITLY, never through the env: an
    # explicit value outranks the env by design, so an arm cannot be steered by
    # an ambient setting the receipt would not show.  The env is CLEARED so a
    # leaked value cannot reach the OFF arms either.
    os.environ.pop(CAPTURE_PIN_ENV, None)
    os.environ.pop(SEAT_NEAR_LIVE_ENV, None)

    repo_dir = Path(tempfile.mkdtemp(prefix=f"grm_rs3_{session_id}_{arm}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    model_info: Any = None
    node_to_idx: dict[str, int] = {}
    live_at_install: list[int] = []
    frame_ephemeral = True
    storage_bits_observed: Any = None
    layer_inventory: dict[str, Any] = {}
    geometry_observed: dict[str, Any] = {}
    try:
        e2e, model, tokenizer, repo, model_info = _load_lived_repo(
            repo_dir, frame, storage_bits="lived")
        arena = repo.arena
        frame_ephemeral = bool(getattr(arena, "ephemeral", False))
        storage_bits_observed = getattr(arena, "storage_bits", None)
        if storage_bits_observed != int(flags["graft_storage_bits"]):
            raise RS3Error(
                f"{arm} must run on the LIVED storage payload "
                f"({flags['graft_storage_bits']} bits) but the arena reports "
                f"storage_bits={storage_bits_observed!r}")
        # THE REGISTERED GEOMETRY IS CHECKED AGAINST THE LIVE ARENA before any
        # arm runs.  A registration describing a different band than the one
        # being measured would invalidate every row silently.
        registered_geometry = registration[
            "geometry_REGISTERED_BEFORE_ANY_GATE"]
        geometry_observed = {
            "n_sink": int(arena.n_sink),
            "arena_width": int(arena.width),
            "live_shift": int(arena.live_shift),
        }
        for key, value in geometry_observed.items():
            if int(registered_geometry[key]) != value:
                raise RS3Error(
                    f"registered {key}={registered_geometry[key]} but the "
                    f"live arena reports {value}: the registration does not "
                    "describe the band being measured")
        layer_inventory = _layer_inventory(arena)
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        live_at_install = [
            int(g) for g, _n in arena.live_segs if g is not None]

        for probe in plan:
            probe_id = str(probe["probe_id"])
            question = str(probe["question"])
            items = [it for it in plan_items if it["probe_id"] == probe_id]
            if not items:
                # A fixture probe with no registered work item is still SERVED
                # through the C0 path, because the state evolution the
                # registered probes inherit has to match C0's.
                started = time.time_ns()
                answer, info, mounts, mass = _serve_ladder(
                    repo, e2e, question, flags)
                info["rs3_arm_note"] = (
                    "not a registered probe: served through the C0 path so "
                    "the repository state this arm's registered probes "
                    "inherit matches C0's")
                served.append(_row(
                    probe, probe_id, session_id, arm, "registered", False,
                    question, answer, mounts, None, None, mass, info,
                    int(arena.cur_mount_n), started, levers))
                continue

            for item in items:
                started = time.time_ns()
                variant = str(item["variant"])
                mount_ids = [int(v) for v in item["mount_ids"]]
                feed_receipt: dict[str, Any] | None = None
                picks_requested: list[int] | None = None

                if arm == "C0" and variant == "registered":
                    answer, info, mounts, mass = _serve_ladder(
                        repo, e2e, question, flags)
                    served.append(_row(
                        probe, probe_id, session_id, arm, variant, True,
                        question, answer, mounts, None, None, mass, info,
                        int(arena.cur_mount_n), started, levers))
                    continue

                # Every other (arm, variant) needs the C0 PRELUDE first.
                prelude = _prelude(repo, e2e, question, flags, arena,
                                   mount_ids)
                base = _snapshot_counterfactual(arena)
                native_base = _snapshot_native_ids(repo)
                missing = [v for v in mount_ids if int(v) >= len(arena.grafts)]
                if missing:
                    raise RS3Error(
                        f"{probe_id}/{arm}/{variant}: mount ids {missing} do "
                        f"not exist after the C0 prelude "
                        f"({len(arena.grafts)} grafts)")
                _clear_boat(arena)

                capture: dict[str, Any] | None = None
                if arm == "C5":
                    feed_receipt = _feed_live(
                        repo, e2e, fixture,
                        [str(v) for v in item["live_node_ids"]])
                    picks_requested = []
                elif arm in CAPTURE_ARMS:
                    capture = _recapture_under_pin(
                        repo, arena, registration, probe_id, arm,
                        levers["capture_pin"], mount_ids)
                    picks_requested = [
                        int(v) for v in capture["fresh_graft_ids"]]
                else:
                    # C0's solace-fact variant and C2: the INSTALLED grafts.
                    picks_requested = list(mount_ids)

                # THE SEATING LEVER, passed explicitly for the duration of this
                # one serve and unwound after.  Explicit beats env by design;
                # the arena attribute is the explicit channel for a caller that
                # cannot reach _rs3_seat_plan's own argument.
                before_seat = getattr(arena, "_rs3_seat_explicit", None)
                try:
                    arena._rs3_seat_explicit = bool(levers["seat_near_live"])
                    answer, info, mounts, mass = _serve_direct(
                        repo, e2e, question, flags, picks_requested)
                finally:
                    arena._rs3_seat_explicit = before_seat

                if capture is not None:
                    info["rs3_capture"] = capture
                info["rs3_c0_prelude"] = prelude
                info["rs3_levers"] = dict(levers)
                served.append(_row(
                    probe, probe_id, session_id, arm, variant, True, question,
                    answer, mounts, picks_requested, feed_receipt, mass, info,
                    int(info.get("rs3_cur_mount_n_at_serve", 0)), started,
                    levers))
                # Leave the arena where the arm found it, so the NEXT probe in
                # the plan inherits C0's state evolution rather than this arm's.
                # The native-id map is rolled back TOO — see
                # _snapshot_native_ids: _restore_counterfactual truncates the
                # graft list but leaves that map, which strands an entry on an
                # index the next probe's fresh graft reuses.
                _restore_counterfactual(arena, base)
                _restore_native_ids(repo, native_base)
    finally:
        try:
            if repo is not None:
                repo.close()
        except Exception:  # noqa: BLE001 - teardown must not mask a result
            pass
        try:
            from core import kv_graft

            if model is not None:
                kv_graft.clear_injection(model)
        except Exception:  # noqa: BLE001
            pass
        del repo, model, tokenizer

    return {
        "schema": f"{SCHEMA_PREFIX}.arm.v1",
        "program": "GRM",
        "phase": "RS3",
        "order": "orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md",
        "arm": arm,
        "levers": levers,
        "session_id": session_id,
        "registered_probe_ids": sorted(targets),
        "frame": {
            "ephemeral": bool(frame_ephemeral),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
        },
        "switches": {
            "GRM_LSR_FIXES": os.environ.get("GRM_LSR_FIXES"),
            "GRM_DEMAND_NGH": os.environ.get("GRM_DEMAND_NGH"),
            CAPTURE_PIN_ENV: os.environ.get(CAPTURE_PIN_ENV),
            SEAT_NEAR_LIVE_ENV: os.environ.get(SEAT_NEAR_LIVE_ENV),
            "levers_passed_explicitly": True,
            "storage_bits": storage_bits_observed,
        },
        "geometry_observed": geometry_observed,
        "arena_width": int(flags["arena_width"]),
        "layer_inventory": layer_inventory,
        "deposit_protocol": (
            "CHRONOLOGICAL_ARENA_FEED (grm_det1_3_gpu._install_lived_nodes), "
            "the LIVED arm; under the SPEC frame that call reaches "
            "ArenaCache.deposit(text) through feed's ephemeral branch"),
        "fixture_node_to_idx": {str(k): int(v)
                                for k, v in node_to_idx.items()},
        "live_graft_ids_after_install": live_at_install,
        "probes": served,
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_rs3_capture_seat_gpu": file_record(Path(__file__).resolve()),
            "grm_rs3_registration": file_record(
                ROOT / "scripts" / "grm_rs3_registration.py"),
            "grm_rs2_mount_read_gpu": file_record(
                ROOT / "scripts" / "grm_rs2_mount_read_gpu.py"),
            "grm_rs1_read_strength_gpu": file_record(
                ROOT / "scripts" / "grm_rs1_read_strength_gpu.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_frame": file_record(ROOT / "core" / "grm_frame.py"),
            "gpt_oss20b_tc": file_record(ROOT / "core" / "gpt_oss20b_tc.py"),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "arm"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--arm", default=None, choices=ARMS)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    registration = read_registration()
    if args.command == "plan":
        sessions = sessions_in_play(registration)
        print(json.dumps({
            "sessions": sessions,
            "arms": list(ARMS),
            "work_items": sum(
                len(arm_plan(registration, s)) for s in sessions),
            "plan": {s: arm_plan(registration, s) for s in sessions},
        }, indent=1))
        return 0

    if not args.session_id:
        raise RS3Error("--session-id is required for the arm command")
    if not args.arm:
        raise RS3Error("--arm is required for the arm command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_arm(str(args.session_id), str(args.arm))
    path = emit(payload, f"grm_rs3_{args.arm}_{args.session_id}")
    print(f"receipt={path}")
    for row in payload["probes"]:
        table = assemble_table([row])[0]
        print(json.dumps({
            "arm": table["arm"],
            "probe_id": table["probe_id"],
            "variant": table["variant"],
            "registered": row["registered_probe"],
            "correct": table["correct"],
            "served": table["served"][:80],
            "mounted_ids": table["mounted_ids"],
            "mounted_full": table["mounted_mass_full_layers"],
            "mounted_sliding": table["mounted_mass_sliding_layers"],
            "live_full": table["live_mass_full_layers"],
            "seat_offset": table["seat_offset_plan_head"],
            "digest": table["graft_digest"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
