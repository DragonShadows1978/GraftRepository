#!/usr/bin/env python3
"""GRM-LSR-P2C.1 G2/G3 — the ten census probes, scored under P2A+P2C+SC1.1.

ORDER: ``orders/GRM_LSR_P2C_1_E2E_CENSUS_SCORING.md``.
REGISTRATION: ``artifacts/lsr_p2c_1/registration.json``.

THE QUESTION.  P2C closed the wrong-value class on the supersession battery.
Its G3 -- the ten ``e2e_t*`` probes of the DET1.11 lived-serving census -- was
BLOCKED because a whole-session INLINE replay reproduced only 8 of 10.  SC1.2
then showed the census's own mechanism is a FORK LADDER, built the matching
instrument, and reproduced 7/7 of the probes it could reach with mass delta
exactly 0.0 -- including t30, one of the two the inline replay lost.  This
driver carries that instrument to all ten and scores the fixes on them.

HOW A PROBE IS MEASURED.  Per probe, one process, one lease:

  1. Reconstruct the routing index AS OF that probe turn from the probe's own
     frozen session repository (READ-ONLY source), truncated to nodes
     ``[0, N)`` with the mutable lineage back-edges RECOMPUTED, written into
     a scratch directory.  N is DERIVED by SC1.2's registered backward walk
     over that session's own instrumentation, never read off the probe id.
  2. Load the model against the reconstruction and verify it holds exactly N
     grafts.
  3. INDEX FIDELITY: re-route the lived question through the reconstruction
     and compare with the ranking the frozen SNAPSHOT recorded.  This is the
     evidence that stands in for the cross-process routing-index verification
     the fork cannot do.
  4. ARM 0 (``GRM_LSR_FIXES=0``, ``GRM_DEMAND_NGH=0``): fork the frozen lived
     snapshot with nothing withheld, continue the prefill, and compare the
     served text against the CENSUS row under the DET1 semantic comparator.
     A probe that does not reproduce is NOT-REPRODUCED and is EXCLUDED from
     Arm 1.  It is never patched.
  5. ARM 1 (``GRM_LSR_FIXES=1``, ``GRM_DEMAND_NGH=0``): the same fork with
     the fixes ON.  Served value, correctness under the DET1 comparator with
     its negative guards, and the fit/grounding receipts.

WHAT THE FIT RECEIPT IS ON THIS PATH, SAID PLAINLY.  ``_continue_forked_prefill``
resumes at the captured prefill boundary: the lived mount set is already
seated in the restored K/V, so the turn does NOT re-enter ``_attempt``'s
admission ladder and ``attach_fit`` never runs.  A fabricated ``fit_*`` block
would therefore be a lie about which code path produced it.  Instead this
driver computes the P2A/P2C fit stage as an explicit COUNTERFACTUAL over the
reconstructed index: ``plan_priority_fit`` is pure arithmetic over
``{node: ntok}`` and the arena's width budget, so the receipt can be derived
from the lived ``admission.rank_plan`` with no forward pass and no state
mutation, and every receipt says ``fit_source =
counterfactual_over_reconstructed_index`` so a reader can never mistake it for
a served-turn receipt.  P2C's split/descent branch is evaluated the same way
and reported whether or not it fires.

THE GROUNDING RECEIPT IS REAL, NOT A COUNTERFACTUAL.  ``_grounding_receipt``
and ``_grounding_attribution`` are pure set arithmetic over already-generated
text and the mounted set, which the fork DOES have.  So the SC1.1 fields
(``grounding_normalized``, ``grounding_glyph_rescued``) and the legacy-vs-
normalized verdict pair are computed on the real Arm-1 answer against the
real Arm-1 mounts.  That is the half of ``GRM_LSR_FIXES`` this instrument can
exercise end to end, and the receipt says which half is which.

FROZEN TREE IS READ-ONLY.  Every path under the frozen run is opened for
reading only.  Reconstructions go to a scratch directory that is removed on
the way out; receipts go to ``artifacts/lsr_p2c_1/``.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on ``/tmp/forge-gpu.lock`` via ``scripts.grm_cmc1_gpu_arms.gpu_lease``;
  * ONE lease per process, <= 520 s, one probe per process;
  * the >= 30 s inter-process gap is taken INSIDE this wrapper BEFORE the
    lease is acquired, so a waiting process never holds the lock while it
    sleeps;
  * the operator has absolute right of way.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402
from scripts import grm_lsr_p2c_1_census as p2c1  # noqa: E402
from scripts import grm_sc1_2_session as sc12  # noqa: E402

ARTIFACT_DIR = p2c1.ARTIFACT_DIR
SCHEMA_PREFIX = "grm.lsr_p2c_1"
ORDER = p2c1.ORDER
REGISTRATION = p2c1.REGISTRATION
RUNTIME_FRAME = p2c1.RUNTIME_FRAME
DET1_REGISTRATION = p2c1.DET1_REGISTRATION

LEASE_SECONDS = 520
LOCK_WAIT_SECONDS = 7200
INTER_RUN_GAP_SECONDS = 30

P2C1Error = p2c1.P2C1Error


def _read(path: Path | str) -> Any:
    return p2c1.read_json(path)


def value_verdict(
    answer: str,
    *,
    expected_values: Sequence[str],
    rejected_values: Sequence[str],
) -> dict[str, Any]:
    """The DET1 unified semantic comparator, with its negative guards.

    Identical to SC1.2's -- imported semantics, not a re-implementation, so a
    value scored correct here is correct by the race's own rule (including
    DET1.10's separator projection, which is what makes the lived
    ``Cobalt 1 India`` a correct ``Cobalt-1-India``).
    """
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


# --------------------------------------------------------------------------
# Index reconstruction on disk
# --------------------------------------------------------------------------
def build_probe_repository(
    probe: Mapping[str, Any], dest: Path,
) -> dict[str, Any]:
    """Write a repository directory holding the index AS OF the probe turn.

    Reads (never writes) the probe's OWN frozen session repository, keeps the
    first ``N`` node payloads and index rows, and writes a manifest whose
    lineage back-edges were recomputed for that prefix by
    ``grm_sc1_2_session.truncated_manifest``.  The WAL directory is created
    EMPTY and the native checkpoint is dropped, so nothing can replay or
    reload a later state behind the truncation.

    THE SOURCE IS THE PROBE'S OWN FAMILY, never a convenient superset: the
    calibration session's repository disagrees with the eval chain's from node
    14 onward, so using one for the other would fabricate an index.
    """
    import numpy as np

    source = p2c1.session_dir(str(probe["session_family"])) / "repository"
    count = int(probe["probe_node_index"])
    source_manifest = _read(source / "manifest.json")
    no_fold_ids = {int(i) for i in probe["no_fold_ids"]}
    manifest = sc12.truncated_manifest(
        source_manifest, count,
        no_fold_at_probe={i: (i in no_fold_ids) for i in range(count)})

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "nodes").mkdir(parents=True, exist_ok=True)
    # An EMPTY wal directory: the manifest's wal_lsn already equals the frozen
    # WAL's max lsn, and shipping no records makes that structural instead of
    # merely true.
    (dest / "wal").mkdir(parents=True, exist_ok=True)

    copied = []
    for index in range(count):
        name = f"{index:04d}.npz"
        src = source / "nodes" / name
        if not src.is_file():
            raise P2C1Error(f"frozen repository lacks node payload {name}")
        shutil.copyfile(src, dest / "nodes" / name)
        copied.append(name)

    with np.load(source / "index.npz") as bundle:
        keys = [f"rkey_{index:04d}" for index in range(count)]
        missing = [k for k in keys if k not in bundle.files]
        if missing:
            raise P2C1Error(f"frozen routing index lacks rows {missing}")
        np.savez(dest / "index.npz", **{k: bundle[k] for k in keys})

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return {
        "schema": f"{SCHEMA_PREFIX}.probe_repository.v1",
        "path": str(dest),
        "node_count": count,
        "session_family": str(probe["session_family"]),
        "source_repository": str(source.relative_to(ROOT)),
        "source_manifest": file_record(source / "manifest.json"),
        "source_index": file_record(source / "index.npz"),
        "node_payloads_copied": len(copied),
        "native_checkpoint_dropped": True,
        "wal_shipped_empty": True,
        "no_fold_ids": sorted(no_fold_ids),
        "no_fold_pinned": bool(probe["no_fold_pinned"]),
        "lineage_rule": (
            "superseded_by / active / retired RECOMPUTED as the closure of "
            "the prefix's own deposit-final `supersedes` edges; never copied "
            "from an end-of-shard flush, which records a LATER state"),
    }


def _open_probe_repo(repo_dir: Path, frame: Mapping[str, Any]):
    """A GraftRepository bound to a reconstructed probe-turn index.

    Every constructor argument matches ``scripts/grm_det1_2_gpu._load_model_repo``
    -- the frozen race's own loader -- except the repository path, which points
    at the reconstruction.  Divergence here would silently change the arena
    the fork is validated against.
    """
    from transformers import AutoTokenizer
    from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_gpu import NATIVE_LIB

    model_dir = str(frame["model"]["path"])
    model, model_info = GptOss20B_TC.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    flags = frame["resolved_flags"]
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
        str(repo_dir),
        autosave=False,
        arena_cls=e2e.GptOssGQAArenaCache,
        native_lib_path=str(NATIVE_LIB),
        native_auto=False,
        vram_budget_mb=None,
        route_layer=int(dialect["route_layer"]),
        arena_width=int(flags["arena_width"]),
        topk=int(flags["topk"]),
        live_turns=int(flags["live_turns"]),
        max_live=int(flags["max_live"]),
        sink_text=e2e.HARMONY_SINK,
        prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS,
        storage_bits=int(flags["graft_storage_bits"]),
        revision_resolution=bool(flags["sup_resolve"]),
        decisive_admission=bool(flags["adm_decisive"]),
    )
    return e2e, model, tokenizer, repo, model_info


# --------------------------------------------------------------------------
# The P2A/P2C fit stage, as an explicit counterfactual
# --------------------------------------------------------------------------
def fit_counterfactual(
    arena, probe: Mapping[str, Any], reachability: Mapping[str, Any],
) -> dict[str, Any]:
    """The P2A + P2C fit receipt for this probe's LIVED plan, derived not run.

    ``_continue_forked_prefill`` resumes at the captured prefill boundary, so
    the turn never re-enters ``_attempt``'s admission ladder and the real
    ``attach_fit`` never fires.  Rather than emit an empty block or invent
    one, this recomputes the fit stage the way the arena would:
    ``plan_priority_fit`` over the lived ``admission.rank_plan``, the lived
    candidate set, this arena's own per-node ``ntok``, and the arena's width
    budget.  It is pure arithmetic -- no forward pass, no routing call, no
    state mutation -- and it is labelled a counterfactual in the receipt so it
    can never be read as a served-turn receipt.

    P2C's split/descent branch is reported through ``fit_unseatable``: a plan
    member whose own ``ntok`` exceeds the whole budget is the ONLY trigger for
    the split, so an empty ``fit_unseatable`` is a positive statement that the
    P2C branch could not fire on this probe, not a silence about it.
    """
    from core.grm_admission import (
        fit_info_fields, plan_priority_fit, split_info_fields,
    )

    rank_plan = [
        int(v) for v in reachability.get("lived_admission_rank_plan", ())]
    ranking = [
        int(v) for v in reachability.get("lived_admission_ranking", ())]
    count = len(arena.grafts)
    candidates = [v for v in ranking if 0 <= v < count]
    ntok = {int(i): int(arena.grafts[i]["ntok"]) for i in candidates}
    # The arena's own budget with no live/recency reservation: the fork
    # restores the lived prefill, whose live window is already inside the
    # captured caches, so the seat budget available to a mount plan is the
    # full arena width.  Stated rather than assumed.
    budget = int(arena.width)
    receipt = plan_priority_fit(
        plan=rank_plan, candidates=candidates, ntok=ntok, budget=budget)
    seated = [int(v) for v in receipt["fit_seated"]]
    head_served = bool(rank_plan) and int(rank_plan[0]) in set(seated)
    fields = fit_info_fields(
        receipt,
        shuttle=bool(receipt.get("fit_shuttle_pending")),
        shuttle_trips=[[v] for v in receipt.get("fit_shuttle_pending", ())],
        served_without_plan_head=bool(rank_plan and not head_served),
    )
    # No split fired: the split is triggered ONLY by an unseatable plan
    # member, and this receipt reports whether there was one.
    fields.update(split_info_fields(
        split_parent=None, split_children=(), split_ephemeral=None,
        descended_head=(), chunk_trips=()))
    fields.update({
        "fit_source": "counterfactual_over_reconstructed_index",
        "fit_source_meaning": (
            "The fork resumes at the captured prefill boundary with the lived "
            "mount set already seated, so _attempt's attach_fit never runs on "
            "this path. plan_priority_fit is pure arithmetic over the lived "
            "rank_plan, this arena's own per-node ntok and the arena width "
            "budget, so the P2A/P2C fit stage is DERIVED here with no forward "
            "pass and no state mutation. It is NOT a served-turn receipt and "
            "must not be read as one."),
        "fit_budget": budget,
        "fit_candidates": candidates,
        "fit_ntok": {str(k): v for k, v in sorted(ntok.items())},
        "fit_shuttle_pending": [
            int(v) for v in receipt.get("fit_shuttle_pending", ())],
        "p2c_split_branch_fired": bool(receipt.get("fit_unseatable")),
        "p2c_split_branch_note": (
            "P2C's split-and-descend fires only when a plan member's own ntok "
            "exceeds the whole budget (fit_unseatable). An empty "
            "fit_unseatable is a positive statement that the branch could not "
            "fire on this probe."),
    })
    return fields


def grounding_receipt(
    arena, *, answer: str, mounted_ids: Sequence[int], question: str,
) -> dict[str, Any]:
    """SC1.1's grounding receipt on the REAL Arm-1 answer and mounts.

    Unlike the fit stage this is not a counterfactual: ``_grounding_receipt``
    and ``_grounding_attribution`` are pure set arithmetic over already-
    generated text and an already-mounted set, both of which the fork has.
    The legacy-vs-normalized verdict pair is computed with an EXPLICIT flag
    (``_grounding_verdict(..., normalized=...)``), never by reading the env
    switch twice, so the counterfactual is real while the switch is ON.
    """
    mounts = [int(v) for v in mounted_ids]
    fields: dict[str, Any] = {}
    grounded, contributors = arena._grounding_attribution(
        answer, mounts, question)
    arena._grounding_receipt(answer, mounts, question, fields)
    legacy, _legacy_contrib = arena._grounding_verdict(
        answer, mounts, question, normalized=False)
    normalized, _norm_contrib = arena._grounding_verdict(
        answer, mounts, question, normalized=True)
    return {
        "grounding_source": "real_answer_real_mounts_no_forward_pass",
        "grounded": bool(grounded),
        "grounding_contributors": sorted(int(v) for v in (contributors or ())),
        "grounding_normalized": bool(fields.get("grounding_normalized")),
        "grounding_glyph_rescued": bool(fields.get("grounding_glyph_rescued")),
        "grounding_verdict_legacy": bool(legacy),
        "grounding_verdict_normalized": bool(normalized),
        "grounding_glyph_projection_changed_the_verdict": (
            bool(normalized) != bool(legacy)),
    }


# --------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------
def _fork_and_generate(
    *, arena, probe, manifest_path, identity, threshold, ngen, fixes_on,
) -> dict[str, Any]:
    """One arm: hydrate from the frozen lived snapshot, generate, observe.

    NOTHING is withheld in either arm.  This order scores SERVING, not the
    demand loop, so there is no planted-miss variant here: the probe under
    test is the lived served control itself.

    ``GRM_LSR_FIXES`` is set EXPLICITLY on both arms.  Its production default
    is ON (it fails closed to ON for an unknown token), so an Arm 0 that
    merely unset it would silently run with the fixes.
    """
    from core import grm_demand
    from scripts.grm_det1_3_snapshot import restore_prefill_fork
    from scripts.grm_det1_4_gpu import _continue_forked_prefill

    os.environ["GRM_LSR_FIXES"] = "1" if fixes_on else "0"
    os.environ["GRM_DEMAND_NGH"] = "0"

    restore = restore_prefill_fork(
        arena,
        Path(manifest_path),
        withheld_mounts=(),
        target_identity=identity,
        # The lived capture process is long gone. This is the DET1.4 zero
        # gate's own path; what it does not verify is recorded by the caller.
        require_same_process_index=False,
    )
    started = time.time_ns()
    # The observer is present purely to RECORD the per-token mounted mass so
    # the Arm-0 mass leg has a fork-side number. Demand is OFF in both arms,
    # so the threshold gates nothing that is measured.
    with grm_demand.DemandObserver(
            arena, int(ngen), float(threshold)) as observer:
        answer, mask_receipt = _continue_forked_prefill(
            arena, restore["prompt_ids"], int(ngen))
    rows = observer.finish()
    elapsed = int(time.time_ns() - started)
    decision = grm_demand.decide(rows, float(threshold))
    mounted_now = [int(v) for v in arena.cur_mounts]
    verdict = value_verdict(
        answer,
        expected_values=probe["expected_values"],
        rejected_values=probe["rejected_values"])

    return {
        "lsr_fixes_env": os.environ.get("GRM_LSR_FIXES"),
        "demand_ngh_env": os.environ.get("GRM_DEMAND_NGH"),
        "demand_loop": "OFF",
        "attempt_answer": str(answer),
        "served_answer": str(answer),
        "served_verdict": verdict,
        "mounted_ids": mounted_now,
        "attempt_min_mass": decision["demand_min_mass"],
        "attempt_token_count": decision["demand_token_count"],
        "elapsed_ns": elapsed,
        "mask_consumption_gate_pass": bool(mask_receipt.get("gate_pass")),
        "fork_intervention": restore["intervention"],
        "fork_source_mounts": restore["source_mounts"],
        "fork_mounts": restore["fork_mounts"],
        "fork_zero_intervention_no_deltas": restore[
            "zero_intervention_no_deltas"],
        "same_process_index_verified": bool(
            restore.get("same_process_index_verified")),
        "routing_index_status": restore.get("routing_index"),
        "source_manifest_sha256": restore["source_manifest_sha256"],
    }


# --------------------------------------------------------------------------
# One probe
# --------------------------------------------------------------------------
def measure_probe(
    *, probe, reachability, frame, threshold, identity, repo, e2e, scratch,
) -> dict[str, Any]:
    """Arm 0 then Arm 1 for one census probe, in that order and deliberately.

    Arm 0 (fixes OFF) runs FIRST so the reproduction verdict is decided before
    anything the fixes do can touch it.  Arm 1 is measured for the record
    either way, but ``arm0_reproduced`` is what the tables read and the
    roll-up excludes a probe that did not reproduce.
    """
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual,
    )

    arena = repo.arena
    flags = frame["resolved_flags"]
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    route_limit = max(topk, (max_trips + 1) * topk)
    manifest_path = ROOT / probe["snapshot_manifest"]
    if not manifest_path.is_file():
        raise P2C1Error(f"frozen lived snapshot absent: {manifest_path}")

    # INDEX-FIDELITY CHECK, before any fork.  Re-route the lived question
    # through the reconstructed index and compare against the ranking the
    # frozen SNAPSHOT recorded for that turn.  This is the evidence that
    # stands in for the cross-process routing-index verification the fork
    # cannot do.
    live_now = {int(g) for g, _n in arena.live_segs if g is not None}
    base = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena, str(probe["question"]),
            (probe["expected_values"] or [""])[0],
            live_excluded=live_now,
            route_limit=route_limit,
        )
    finally:
        _restore_counterfactual(arena, base)
    reconstructed = [int(v) for v in profile.get("raw_ranking", ())]
    lived_ranking = [
        int(v) for v in reachability.get("lived_admission_ranking", ())]
    index_fidelity = {
        "lived_admission_ranking": lived_ranking,
        "reconstructed_raw_ranking_ids": reconstructed,
        "rank1_match": bool(
            reconstructed and lived_ranking
            and reconstructed[0] == lived_ranking[0]),
        "prefix_match": bool(
            reconstructed[:len(lived_ranking)] == lived_ranking),
        "node_count": int(probe["probe_node_index"]),
        "meaning": (
            "The reconstructed probe-turn index must route the lived question "
            "the way the lived turn did. rank1_match is the load-bearing leg "
            "(it is the node the turn mounts); prefix_match is reported "
            "alongside and a tail difference is named, not hidden."),
    }

    # -- ARM 0: fixes OFF, demand OFF ------------------------------------
    arm0 = _fork_and_generate(
        arena=arena, probe=probe, manifest_path=manifest_path,
        identity=identity, threshold=threshold, ngen=ngen, fixes_on=False)
    arm0.update(p2c1.reproduction_verdict(
        lived_answer=probe["lived"]["served_answer"],
        fork_answer=arm0["attempt_answer"],
        lived_min_mass=probe["lived"]["served_min_mass"],
        fork_min_mass=arm0["attempt_min_mass"],
        mass_leg_available=bool(probe["mass_leg_available"]),
        # A zero-intervention fork is byte-identical to the lived capture's
        # arrays, which is what licenses exact float equality on the mass leg.
        byte_exact_fork=bool(arm0["fork_zero_intervention_no_deltas"]),
    ))

    # -- ARM 1: fixes ON, demand OFF -------------------------------------
    arm1 = _fork_and_generate(
        arena=arena, probe=probe, manifest_path=manifest_path,
        identity=identity, threshold=threshold, ngen=ngen, fixes_on=True)
    # The fixes must be ON while the grounding receipt is taken: that is the
    # half of GRM_LSR_FIXES this instrument exercises for real.
    arm1["grounding_receipt"] = grounding_receipt(
        arena,
        answer=arm1["served_answer"],
        mounted_ids=arm1["mounted_ids"],
        question=str(probe["question"]))
    arm1["fit_receipt"] = fit_counterfactual(arena, probe, reachability)
    # Reported alongside Arm 0's own verdict: did the FIXES change the served
    # text at all on this probe? A no-change row is a result, not a null.
    arm1["arm1_text_differs_from_arm0"] = (
        str(arm1["served_answer"]) != str(arm0["served_answer"]))

    return {
        "probe_id": str(probe["probe_id"]),
        "session_family": str(probe["session_family"]),
        "conversation_turn": int(probe["conversation_turn"]),
        "probe_node_index": int(probe["probe_node_index"]),
        "role": probe["role"],
        "split": probe["split"],
        "spec": probe["spec"],
        "census_attempt": probe["census_attempt"],
        "census_class": probe["census_class"],
        "census_subclass": probe["census_subclass"],
        "separator_artifact_rescued": bool(
            probe["separator_artifact_rescued"]),
        "question": str(probe["question"]),
        "expected_values": list(probe["expected_values"]),
        "rejected_values": list(probe["rejected_values"]),
        "no_fold_ids": list(probe["no_fold_ids"]),
        "no_fold_pinned": bool(probe["no_fold_pinned"]),
        "mass_leg_available": bool(probe["mass_leg_available"]),
        "session_resume_path": "a_frozen_per_probe_turn_snapshot_fork",
        "index_reconstruction": dict(scratch),
        "index_fidelity": index_fidelity,
        "reachability": dict(reachability),
        "frozen_snapshot": file_record(manifest_path),
        "lived": dict(probe["lived"]),
        "arm0_reproduced": bool(arm0["reproduced"]),
        "arm0": arm0,
        "arm1": arm1,
    }


# --------------------------------------------------------------------------
# Probe plan (CPU) and the leased run
# --------------------------------------------------------------------------
def probe_plan() -> dict[str, Any]:
    """Every probe with its family, derived N, reachability -- CPU only."""
    families = sorted(set(p2c1.PROBE_SESSIONS.values()))
    node_counts = {f: p2c1.family_probe_node_counts(f) for f in families}
    flushes = {f: p2c1.no_fold_flushes(f) for f in families}
    probes = p2c1.census_probes(node_counts=node_counts, flushes=flushes)
    rows = [
        {**probe, "reachability": p2c1.snapshot_reachability(probe)}
        for probe in probes
    ]
    return {
        "schema": f"{SCHEMA_PREFIX}.plan.v1",
        "probe_count": len(rows),
        "families": {
            family: {
                "session_dir": p2c1.SESSION_FAMILIES[family],
                "probe_node_counts": {
                    str(k): v for k, v in node_counts[family].items()},
                "no_fold_flushes": [
                    {"node_count": count, "no_fold_ids": list(ids)}
                    for count, ids in flushes[family]
                ],
            }
            for family in families
        },
        "probes": rows,
    }


def reconstruction_proofs(family: str) -> dict[str, Any]:
    """The two fail-closed proofs the index reconstruction rests on.

    (1) LINEAGE: recomputing ``superseded_by`` / ``active`` / ``retired`` from
        a prefix's own forward edges must reproduce every frozen flush in this
        family at its own node count.  Zero mismatches or the run aborts.
    (2) PREFIX AGREEMENT: every flush in the family must agree with the
        family's widest manifest on the deposit-time fields the
        reconstruction actually reads.  Any disagreement means the
        single-source reconstruction is invalid.
    """
    manifests = p2c1.family_flush_manifests(family)
    lineage = sc12.lineage_selfcheck(manifests)
    widest = max(manifests.values(), key=lambda m: len(m.get("nodes") or ()))
    agreements = {
        shard: sc12.prefix_agreement(widest["nodes"], manifest["nodes"])
        for shard, manifest in manifests.items()
    }
    return {
        "family": family,
        "flush_count": len(manifests),
        "flush_node_counts": {
            shard: len(manifest.get("nodes") or ())
            for shard, manifest in manifests.items()
        },
        "lineage_selfcheck": lineage,
        "prefix_agreement": agreements,
        "all_agree": all(a["agree"] for a in agreements.values()),
    }


def run_probe(probe_id: str) -> dict[str, Any]:
    """Reconstruct one probe's index, load the model once, run both arms."""
    from core import grm_demand
    from scripts.grm_det1_3_gpu import _frame_identity
    from scripts.grm_det1_5_gpu import _process_instance
    from scripts.grm_det1_gpu import (
        _runtime_env, _validate_visibility, NATIVE_LIB,
    )

    frame = _read(RUNTIME_FRAME)
    if frame.get("native_library") != file_record(NATIVE_LIB):
        raise P2C1Error("native runtime drifted from the frozen frame")
    os.environ.update(_runtime_env(frame))
    visible = _validate_visibility()
    if str(visible) != str(frame["cuda_visible_devices"]):
        raise P2C1Error("visible GPU differs from the frozen runtime frame")

    threshold = grm_demand.registered_threshold()
    registered = grm_demand.load_registered()

    plan = probe_plan()
    probes = {row["probe_id"]: row for row in plan["probes"]}
    if probe_id not in probes:
        raise P2C1Error(f"{probe_id} is not one of the ten census probes")
    probe = probes[probe_id]
    reachability = probe["reachability"]
    if not reachability.get("reachable"):
        # UNREACHABLE is a RESULT: it is emitted as its own receipt and no
        # measurement is attempted, rather than being forced through.
        return {
            "schema": f"{SCHEMA_PREFIX}.probe_measurement.v1",
            "gate": "G2+G3",
            "probe_id": probe_id,
            "status": "UNREACHABLE",
            "order": file_record(ORDER),
            "registration": file_record(REGISTRATION),
            "result": {
                "probe_id": probe_id,
                "session_family": probe["session_family"],
                "conversation_turn": probe["conversation_turn"],
                "probe_node_index": probe["probe_node_index"],
                "reachability": reachability,
                "lived": probe["lived"],
                "arm0_reproduced": None,
                "arm0": {},
                "arm1": {},
            },
        }

    family = str(probe["session_family"])
    proofs = reconstruction_proofs(family)
    if not proofs["lineage_selfcheck"]["all_match"]:
        raise P2C1Error(
            f"lineage derivation does not reproduce the frozen {family} "
            "manifests; the index reconstruction is not trustworthy")
    if not proofs["all_agree"]:
        raise P2C1Error(
            f"frozen {family} manifests disagree on deposit-time fields")

    work = Path(tempfile.mkdtemp(prefix=f"lsr_p2c_1_{probe_id}_"))
    repo = model = tokenizer = None
    model_info: Any = None
    try:
        scratch = build_probe_repository(probe, work / "repository")
        e2e, model, tokenizer, repo, model_info = _open_probe_repo(
            work / "repository", frame)
        observed = len(repo.arena.grafts)
        if observed != int(probe["probe_node_index"]):
            raise P2C1Error(
                f"reconstructed repository loaded {observed} nodes, expected "
                f"{probe['probe_node_index']}")
        identity = _frame_identity(
            model, tokenizer, e2e, _read(DET1_REGISTRATION), frame)
        process = _process_instance()
        result = measure_probe(
            probe=probe, reachability=reachability, frame=frame,
            threshold=threshold, identity=identity, repo=repo, e2e=e2e,
            scratch=scratch)
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
        shutil.rmtree(work, ignore_errors=True)

    return {
        "schema": f"{SCHEMA_PREFIX}.probe_measurement.v1",
        "gate": "G2+G3",
        "probe_id": probe_id,
        "status": "MEASURED",
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "census": file_record(p2c1.CENSUS),
        "candidate_observations": file_record(p2c1.CANDIDATES),
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": str(registered["caveat"]),
        "carried_threshold_role_here": (
            "DIAGNOSTIC ONLY. GRM_DEMAND_NGH is OFF in BOTH arms, so the "
            "threshold gates nothing that is measured; it is passed to the "
            "DemandObserver purely so the per-turn minimum mounted mass is "
            "RECORDED for the Arm-0 mass leg."),
        "session_resume_protocol": (
            "The certified session is RESUMED at the probe turn by hydrating "
            "the arena from the frozen per-probe-turn lived snapshot with "
            "restore_prefill_fork -- the same DET1.3 mechanism that produced "
            "the lived rows. No session replay. The campaign worker's "
            "_context is never called and is not pointed at a copy: this "
            "driver imports the measurement functions instead."),
        "why_this_instrument_and_not_P2C_G3s": (
            "The DET1.11 census rows were produced by a COUNTERFACTUAL FORK "
            "LADDER, not by an inline turn path -- P2C's own G3 blocked "
            "report diagnosed exactly that. Verified per probe here: the "
            "snapshot's linked_answer records probe_selected_attempt 0 and "
            "its probe_answer is byte-equal to the census served_answer, so "
            "the lived text was served at ladder rung 0, which is the rung "
            "this snapshot captures."),
        "frozen_tree_write_policy": (
            "READ-ONLY. Every path under the frozen run is opened for reading "
            "only; reconstructions and receipts are written elsewhere."),
        "cross_process_fork_caveat": (
            "require_same_process_index=False (the lived process is gone). "
            "This is the DET1.4 zero gate's own path. restore() therefore "
            "reports routing_index=NOT_VERIFIED_CROSS_PROCESS_ZERO_GATE_ONLY: "
            "the full D-LQR index projection is NOT proven identical to the "
            "lived process's. D-LQR is REFUTED-STRUCTURAL and unused here; "
            "the index fidelity this measurement depends on is evidenced by "
            "index_fidelity and by the Arm 0 reproduction itself."),
        "reconstruction_proofs": proofs,
        "family_probe_node_counts": plan["families"][family][
            "probe_node_counts"],
        "process": process,
        "result": result,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_lsr_p2c_1_gpu": file_record(Path(__file__).resolve()),
            "grm_lsr_p2c_1_census": file_record(
                ROOT / "scripts" / "grm_lsr_p2c_1_census.py"),
            "grm_sc1_2_session": file_record(
                ROOT / "scripts" / "grm_sc1_2_session.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "grm_text_norm": file_record(ROOT / "core" / "grm_text_norm.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "grm_det1_3_snapshot": file_record(
                ROOT / "scripts" / "grm_det1_3_snapshot.py"),
        },
    }


def summarize(receipts: Sequence[Path]) -> dict[str, Any]:
    """The G2 reproduction table and the G3 scoring table."""
    results = []
    for path in receipts:
        payload = _read(Path(path))
        results.append({**payload["result"], "_receipt": str(path)})
    order = {pid: i for i, pid in enumerate(p2c1.CENSUS_PROBE_IDS)}
    results.sort(key=lambda r: order.get(str(r["probe_id"]), 99))

    return {
        "schema": f"{SCHEMA_PREFIX}.summary.v1",
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "census": file_record(p2c1.CENSUS),
        "G2_arm0_reproduction": p2c1.g2_table(results),
        "G3_arm1_scoring": p2c1.g3_table(results),
        "probe_receipts": [file_record(Path(p)) for p in receipts],
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = ARTIFACT_DIR / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "probe", "summarize"))
    parser.add_argument("--probe-id", default=None)
    parser.add_argument("--receipts", nargs="*", default=())
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    parser.add_argument("--gap-seconds", type=int,
                        default=INTER_RUN_GAP_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        plan = probe_plan()
        print(json.dumps({
            "probe_count": plan["probe_count"],
            "families": plan["families"],
            "probes": [
                {
                    "probe_id": p["probe_id"],
                    "session_family": p["session_family"],
                    "conversation_turn": p["conversation_turn"],
                    "probe_node_index": p["probe_node_index"],
                    "reachable": p["reachability"]["reachable"],
                    "unreachable_reasons": p["reachability"]["reasons"],
                    "mass_leg_available": p["mass_leg_available"],
                    "no_fold_ids": p["no_fold_ids"],
                    "no_fold_pinned": p["no_fold_pinned"],
                    "lived_answer": p["lived"]["served_answer"],
                    "lived_correct": p["lived"]["served_correct"],
                    "lived_min_mass": p["lived"]["served_min_mass"],
                }
                for p in plan["probes"]
            ],
        }, indent=1))
        return 0

    if args.command == "summarize":
        payload = summarize([Path(p) for p in args.receipts])
        path = emit(payload, "lsr_p2c_1_summary")
        g2 = payload["G2_arm0_reproduction"]
        g3 = payload["G3_arm1_scoring"]
        print(f"receipt={path}")
        print(f"G2 reachable={g2['reachable_count']}/{g2['probe_count']} "
              f"reproduced={g2['reproduced_count']}/{g2['probe_count']} "
              f"gate_pass={g2['gate_pass']}")
        for row in g2["table"]:
            print("G2 " + json.dumps(row))
        print(f"G3 measured={g3['arm1_measured_count']} "
              f"lived_correct_measured={g3['lived_correct_measured_count']} "
              f"regressions={g3['regression_count']} "
              f"fixes={g3['fix_count']} gate_pass={g3['gate_pass']}")
        for row in g3["table"]:
            print("G3 " + json.dumps({
                key: row[key] for key in (
                    "probe_id", "lived_value", "lived_verdict",
                    "arm0_reproduced", "arm1_value", "arm1_correct", "change")
            }))
        return 0 if (g2["gate_pass"] and g3["gate_pass"]) else 1

    if not args.probe_id:
        raise P2C1Error("--probe-id is required for the probe command")
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    # The >= 30 s inter-process gap, taken INSIDE the wrapper BEFORE the lease
    # is acquired, so a waiting process never holds the lock while it sleeps.
    if int(args.gap_seconds) > 0:
        time.sleep(int(args.gap_seconds))
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_probe(str(args.probe_id))
    path = emit(payload, f"lsr_p2c_1_probe_{args.probe_id}")
    result = payload["result"]
    print(f"receipt={path}")
    if payload["status"] == "UNREACHABLE":
        print(json.dumps({
            "probe_id": result["probe_id"],
            "status": "UNREACHABLE",
            "reasons": result["reachability"]["reasons"],
        }))
        return 0
    arm0, arm1 = result["arm0"], result["arm1"]
    print(json.dumps({
        "probe_id": result["probe_id"],
        "session_family": result["session_family"],
        "probe_node_index": result["probe_node_index"],
        "index_rank1_match": result["index_fidelity"]["rank1_match"],
        "index_prefix_match": result["index_fidelity"]["prefix_match"],
        "lived_answer": result["lived"]["served_answer"],
        "lived_correct": result["lived"]["served_correct"],
        "arm0_answer": arm0["served_answer"],
        "arm0_reproduced": arm0["reproduced"],
        "arm0_text_normalized_match": arm0["text_normalized_match"],
        "arm0_text_bytes_match": arm0["text_bytes_match"],
        "mass_leg_available": arm0["mass_leg_available"],
        "arm0_lived_min_mass": arm0["lived_min_mass"],
        "arm0_fork_min_mass": arm0["fork_min_mass"],
        "arm0_min_mass_delta": arm0["min_mass_delta"],
        "arm1_answer": arm1["served_answer"],
        "arm1_correct": arm1["served_verdict"]["correct"],
        "arm1_mounted_ids": arm1["mounted_ids"],
        "arm1_text_differs_from_arm0": arm1["arm1_text_differs_from_arm0"],
        "arm1_grounding": arm1["grounding_receipt"],
        "arm1_fit": {
            key: arm1["fit_receipt"][key] for key in (
                "fit_planned", "fit_seated", "fit_dropped_planned",
                "fit_unseatable", "served_without_plan_head",
                "p2c_split_branch_fired", "fit_source")
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
