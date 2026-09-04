#!/usr/bin/env python3
"""GRM-RS1 — read-strength probe on the three spec-frame refusers.

ORDER: ``orders/GRM_RS1_READ_STRENGTH_PROBE.md``.
REGISTRATION: ``artifacts/grm_rs1/registration.json`` (written FIRST, by
``scripts/grm_rs1_registration.py``; this module refuses to run without it).

THE QUESTION.  EB1 made the ephemeral boat the production default.  Under it
three sup probes REFUSE with the correct node mounted — identical
``mounted_ids`` to the lived run, and ``live_graft_ids_after_install`` went
[2, 3] -> [].  The live window, not the mounted graft, had been carrying those
reads.  This module measures WHAT the live window contributed that the graft
alone does not.  MEASUREMENT ONLY: nothing under ``core/`` or ``config/`` is
touched, no threshold is retuned, no separator is folded, and no production
path changes.

THE BUILD, and why it is the P2C build and not a new one.  Every arm starts
from the SAME repository construction the EB1 G2 rows came from:
``grm_det1_2_gpu._load_model_repo`` (arena_width 96, topk 3, live_turns 2,
storage_bits 8, A-DEC on) followed by
``grm_det1_3_gpu._install_lived_nodes`` (chronological ``arena.feed``, every
node pinned ``kind="fact"``).  Both are IMPORTED from the read-only lived
modules; neither is reimplemented here.  ``scripts/lsr_p2c_replay_gpu`` is the
frozen template this module follows, and its serving core
(``e2e._probe_ladder_chat`` with ``defer_memory=True``) is what A0 calls.

PROBE ORDER MATTERS, and A0 preserves it.  ``_run_sup`` serves a fixture's
probes SEQUENTIALLY against one repository, and the arena is left in the
served rung's state — that is why probe 1 of ``fresh_fact_controls`` routes
over a live-excluded repository and probe 2 routes over the whole one.  A0
therefore replays the fixture's WHOLE probe plan in the lived order and reads
the target rows out of it; serving a target probe alone would be a different
computation and would not be an EB1 reproduction.  Non-target rows stay in the
receipt so a reader can see the state a target probe inherited.

THE FIVE ARMS, and the exact seam each one needs.

  A0  spec frame, graft only.  The production probe path, unchanged.

  A1  the crutch isolated: the registered ``live_ids`` node TEXT pushed
      through the LIVE cache, and NO graft mounted.
      SEAM: under the spec frame ``ArenaCache.feed`` is a no-op that deposits
      instead of pushing (``if self.ephemeral: ... return``), and
      ``eb1_begin_turn`` clears the boat at the top of every served turn.  So
      a live-band read cannot be produced through the spec-frame turn-open at
      all.  This arm therefore uses the PERSISTENT-frame feed mechanics — the
      arena's ``ephemeral`` attribute is pinned False for the duration of the
      FEED ONLY, which is exactly what ``GRM_PERSISTENT_BOAT`` selects and what
      every persistent-frame receipt in this repository was taken under — and
      then serves through ``arena._attempt`` with an EMPTY pick list.  Calling
      ``_attempt`` directly is what the ladder does at its bottom; this arm
      skips the ladder because the arm's whole definition is "routing
      disabled", and a routed rung would re-introduce the mount the arm exists
      to remove.  Both deviations are stated on the receipt.

  A2  the same node(s) as A1, MOUNTED alongside A0's mount, no live window.
      SEAM: a direct ``arena._attempt(question, picks, ...)`` with
      ``picks = A0's mounted_ids | live_ids``.  Same serving call as A1, so the
      two arms differ ONLY in which band the payload sits in — which is
      precisely the CONTENT-vs-POSITION contrast the order asks for.

  A3  unquantized graft.  SEAM: ``_load_model_repo`` hard-codes
      ``storage_bits=8`` and is READ-ONLY under this order, so this module
      builds the repository itself with ``storage_bits=None`` (the fp16 node
      payload).  Every other constructor argument is read from the FROZEN
      runtime frame's ``resolved_flags`` — the same source
      ``grm_lsr_p2c_1_gpu._open_probe_repo`` uses for the same reason — so the
      only thing that differs from the lived build is the one varied argument.
      The receipt records the arena's OBSERVED ``storage_bits`` and the arm
      raises if it is not the fp16 payload.

  A4  instruct-prior control.  SEAM: the QUESTION TEXT is rewritten in the
      repo's registered strict forced-final wording
      (``scripts/gpt_oss20b_turn_prompt_sweep.py`` 'turn50_strict', the wording
      ``docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`` records as the one that beat
      the prior at 64K where the generic turn-50 wordings lost).  The probe's
      subject phrase — and with it every identifier token routing and
      admission key on — is preserved verbatim; only the frame around it
      changes.  Both wordings are on the receipt.

  A5  mass partition on every arm.  SEAM: ``core.grm_demand.DemandObserver``
      already computes exactly the four-way partition the order asks for
      (physical sink / mounted band / live band / learned sink), per FULL
      ATTENTION LAYER, and then averages the layers away in
      ``_summarize_mass``.  ``LayerMassObserver`` below wraps it and retains
      the per-layer rows it was already computing.  NOTHING in the computation
      changes — the wrapper keeps a copy and delegates — so the mounted-mass
      numbers are the ones production reports.  ``GRM_DEMAND_NGH`` stays OFF
      for every arm: the observer is attached by this harness for measurement,
      never gates, and never fires a demand trip.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via ``grm_cmc1_gpu_arms.gpu_lease``; the
    operator has ABSOLUTE right of way and this module never signals, kills, or
    interrupts any process — it waits on the flock or reports blocked;
  * ONE arm over ONE fixture per process, one lease per process, under the
    house cap; the caller inserts the inter-process gap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_frame import ENV_NAME as PERSISTENT_BOAT_ENV  # noqa: E402
from core.grm_frame import env_persistent_boat  # noqa: E402
from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes)
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402
from scripts.grm_rs1_registration import RS1Error  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs1"
REGISTRATION = ARTIFACT_DIR / "registration.json"
SCHEMA_PREFIX = "grm.rs1"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 560
LOCK_WAIT_SECONDS = 7200

#: Every arm this module knows how to run, in registration order.
#:
#: A2 SPLITS INTO TWO SUB-ARMS, and the reason is a MEASURED width fact rather
#: than a preference.  The order defines A2 as "the same node(s) as A1 but
#: mounted as grafts ALONGSIDE A0's mount".  Measured on this battery at the
#: lived ``arena_width`` of 96: the fixture nodes run 52-66 tokens each and the
#: competitor nodes 150-159, so NO TWO of them co-seat and a competitor does
#: not seat at all.  The literal union is therefore unseatable, and production's
#: own packer (``_budget_fit_mounts``, rank order) resolves it by keeping A0's
#: mount and dropping the restatement — which makes the literal arm a byte-copy
#: of A0 and answers nothing.
#:
#:   A2a  the LITERAL arm: A0's mount UNION the registered live ids, packed by
#:        production's fit.  Run and reported because "the union does not fit"
#:        is itself the finding, and the receipt names what the fit dropped.
#:   A2b  the arm the order's own sentence asks for: the registered live-id
#:        node(s) mounted ALONE, in place of A0's mount.  This is A1's payload
#:        moved from the LIVE band to the MOUNT band with nothing else changed,
#:        which is exactly the CONTENT-vs-POSITION contrast A2 exists to draw.
#:
#: The G3 verdict rule reads "A2 rescues" as EITHER sub-arm rescuing: a mount
#: of the restatement that serves correctly is content doing the work, however
#: the mount set was assembled.
ARMS = ("A0", "A1", "A2a", "A2b", "A3", "A4")

#: The sub-arms that stand for the order's A2.
A2_ARMS = ("A2a", "A2b")

#: The arms that serve through ``arena._attempt`` with a hand-built mount set.
DIRECT_ARMS = ("A1", *A2_ARMS)

#: The four masses the D-NGH partition is made of, in the observer's order.
MASS_NAMES = (
    "physical_sink_mass", "mounted_mass", "live_mass", "learned_sink_mass")


# ======================================================================
# A5 — the instrument
# ======================================================================
class LayerMassObserver:
    """The production D-NGH observer, with its per-layer rows RETAINED.

    ``core.grm_demand.DemandObserver`` computes one four-way partition row per
    FULL ATTENTION LAYER per token and then averages the layers away in
    ``_summarize_mass``.  Everything A5 asks for beyond the mean — the
    per-layer partition and the top contributing layer — is already in those
    rows; they are simply discarded.  This wrapper keeps a copy.

    IT CHANGES NO ARITHMETIC.  ``_full_mass`` is untouched, the wrapped
    ``_summarize_mass`` is called and its return value passed straight back,
    and the partition-sums-to-one check the base class enforces still runs.
    The only added behaviour is the append — and it happens BEFORE the
    delegation, because that check RAISES, and a raise must not cost us the
    rows that show why.
    """

    def __init__(self, arena: Any, ngen: int, threshold: float) -> None:
        from core import grm_demand

        # Composition rather than subclassing: DemandObserver is a dataclass
        # whose __post_init__ owns the instrument state, and wrapping keeps
        # every one of those invariants the base class's own.
        self._inner = grm_demand.DemandObserver(
            arena, int(ngen), float(threshold), early_abort=False)
        self.layer_rows: list[list[dict[str, float]]] = []
        original = self._inner._summarize_mass

        def summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
            self.layer_rows.append([
                {str(k): float(v) for k, v in dict(row).items()}
                for row in rows
            ])
            return original(rows)

        self._inner._summarize_mass = summarize  # type: ignore[method-assign]

    def __enter__(self) -> "LayerMassObserver":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def rows(self) -> list[dict[str, Any]]:
        """The answer-position rows, under the race's flush-drop convention."""
        return self._inner.finish()

    def partition(self) -> dict[str, Any]:
        """The A5 table for ONE serve.

        ``mean_over_answer_positions`` is the four-way partition averaged over
        the answer positions the race keeps.  ``min_mounted_mass`` is the same
        statistic ``grm_demand.decide`` reports, computed the same way, so the
        number here and the number a production receipt would carry are one
        number.  ``top_contributing_layer`` is the full-attention layer ORDINAL
        (index into the observed full-layer sequence, which the observer walks
        in model order) whose mounted mass is highest, averaged over those same
        answer positions.
        """
        kept = self.rows()
        # The layer rows were appended once per FORWARD; the kept rows are the
        # ANSWER positions after the flush-drop. Align by truncating the layer
        # sequence to the kept length, which is what the drop did to the rows.
        layers = self.layer_rows[:len(kept)]
        mean = {
            name: (float(np.mean([float(r[name]) for r in kept]))
                   if kept else None)
            for name in MASS_NAMES
        }
        per_layer: list[dict[str, Any]] = []
        if layers:
            width = len(layers[0])
            for ordinal in range(width):
                row: dict[str, Any] = {
                    name: float(np.mean(
                        [float(step[ordinal][name]) for step in layers]))
                    for name in MASS_NAMES
                }
                row["full_layer_ordinal"] = int(ordinal)
                per_layer.append(row)
        top = (
            max(per_layer, key=lambda r: float(r["mounted_mass"]))
            if per_layer else None)
        return {
            "answer_positions": len(kept),
            "full_attention_layers": (len(layers[0]) if layers else 0),
            "mean_over_answer_positions": mean,
            "min_mounted_mass": (
                min(float(r["mounted_mass"]) for r in kept) if kept else None),
            "max_mounted_mass": (
                max(float(r["mounted_mass"]) for r in kept) if kept else None),
            "per_token_mounted_mass": [float(r["mounted_mass"]) for r in kept],
            "per_token_live_mass": [float(r["live_mass"]) for r in kept],
            "per_layer_mean": per_layer,
            "top_contributing_layer": (
                None if top is None else {
                    "full_layer_ordinal": int(top["full_layer_ordinal"]),
                    "mounted_mass": float(top["mounted_mass"]),
                }),
            "partition_sum": (
                None if not kept
                else float(sum(float(v) for v in mean.values()))),
            "partition_sums_to_one": (
                None if not kept else bool(partition_sum_ok(mean))),
        }


def partition_sum_ok(mean: Mapping[str, Any]) -> bool:
    """The partition arithmetic check, exposed for the CPU tests.

    Four masses summing to one within the same tolerance
    ``grm_demand._summarize_mass`` enforces.  A ``None`` in the mapping means
    "no answer positions were captured", which is not a partition at all.
    """
    values = [mean.get(name) for name in MASS_NAMES]
    if any(value is None for value in values):
        return False
    return 0.98 <= float(sum(float(v) for v in values)) <= 1.02


# ======================================================================
# Registration reading (pure — CPU tested)
# ======================================================================
def read_registration(path: Path = REGISTRATION) -> dict[str, Any]:
    if not Path(path).exists():
        raise RS1Error(
            f"registration missing at {path}. It is written BEFORE any gate "
            "run by scripts/grm_rs1_registration.py; this harness refuses to "
            "measure anything without it.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registered_probes(registration: Mapping[str, Any]) -> dict[str, Any]:
    return dict(registration["probes_REGISTERED_BEFORE_ANY_GATE"])


def probes_for_session(registration: Mapping[str, Any],
                       session_id: str) -> list[str]:
    """The registered probe ids bound to one fixture, in registration order."""
    return [
        probe_id
        for probe_id, row in registered_probes(registration).items()
        if str(row["session_id"]) == str(session_id)
    ]


def sessions_in_play(registration: Mapping[str, Any]) -> list[str]:
    """Every fixture at least one registered probe belongs to."""
    seen: list[str] = []
    for row in registered_probes(registration).values():
        session = str(row["session_id"])
        if session not in seen:
            seen.append(session)
    return seen


def arm_plan(registration: Mapping[str, Any],
             session_id: str) -> list[dict[str, Any]]:
    """The (arm, probe) work items for one fixture, in arm-then-probe order.

    Pure: no GPU, no model, no filesystem beyond the registration already
    read.  This is the function G1's CPU tests exercise.
    """
    probes = registered_probes(registration)
    ids = probes_for_session(registration, session_id)
    if not ids:
        raise RS1Error(f"no registered probe belongs to fixture {session_id}")
    plan: list[dict[str, Any]] = []
    for arm in ARMS:
        for probe_id in ids:
            row = probes[probe_id]
            plan.append({
                "arm": arm,
                "probe_id": probe_id,
                "session_id": session_id,
                "role": str(row["role"]),
                "a0_mounted_ids": [int(v) for v in row["a0_mounted_ids"]],
                "live_ids": [int(v) for v in row["live_ids"]],
                "live_node_ids": [str(v) for v in row["live_node_ids"]],
            })
    return plan


# ======================================================================
# The A4 wording
# ======================================================================
#: The repo's registered strict forced-final wording, quoted from
#: ``scripts/gpt_oss20b_turn_prompt_sweep.py`` ('turn50_strict') — the variant
#: docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md records as the one that returned the
#: stored value as top-1 where the generic turn-50 wordings returned "The".
#:
#: THE LEVER IS THE EXTRACTION INSTRUCTION, NOT THE TURN LABEL, and this arm
#: carries only the lever.  Measured on the first A4 run, with a receipt: the
#: literal ``"Turn 50. Previous context is cleared. ..."`` prefix plants the
#: numeral ``50`` in the query, ``_probe_identifier_tokens`` reads it as an
#: IDENTIFIER, nothing in the repository binds it, and P2A's
#: ``identifier_unbound`` abstention fires BEFORE any mount work — every probe
#: served ``"Not in memory: no stored record matches 50."`` with an EMPTY
#: mount and no mass rows at all.  That measures the abstention rule, not the
#: instruct prior, so it is a harness artifact and no verdict rests on it.
#:
#: The sweep's own finding is that the turn label is the part that HURT (the
#: turn-labelled variants returned ``The``; the strict extraction instruction
#: is what recovered the value), so dropping the label keeps the lever the arm
#: is testing and removes the confound. ``STRICT_TURN_PREFIX`` is kept here so
#: the receipt and the tests can name exactly what was dropped and why.
STRICT_TURN_PREFIX = "Turn 50. Previous context is cleared. "
STRICT_PREFIX = "Return exactly the "
STRICT_SUFFIX = ". No sentence. No punctuation."
_QUESTION_HEAD = "what is the "


def strict_wording(question: str) -> str:
    """Rewrite a sup probe question in the registered strict wording.

    The sup questions are all "What is the current <SUBJECT> value?".  The
    rewrite keeps "current <SUBJECT> value" — and with it every identifier
    token routing and admission key on — and replaces only the frame around
    it.  A question outside that family is wrapped in the same strict frame
    with its whole text as the subject, so the arm never silently skips one.

    NO NUMERAL IS INTRODUCED.  See ``STRICT_TURN_PREFIX`` above: the sweep's
    turn label plants an unbindable identifier and short-circuits the probe
    into P2A's abstention, which would measure the abstention rule instead of
    the prior.
    """
    text = str(question).strip()
    subject = text
    if text.lower().startswith(_QUESTION_HEAD):
        subject = text[len(_QUESTION_HEAD):].rstrip("?").strip()
    return f"{STRICT_PREFIX}stored {subject}{STRICT_SUFFIX}"


# ======================================================================
# The verdict helpers (pure — CPU tested)
# ======================================================================
def answer_verdict(answer: str, *, expected_values: Sequence[str],
                   rejected_values: Sequence[str]) -> dict[str, Any]:
    """The DET1 unified semantic comparator with its negative guards.

    Identical in shape to ``lsr_p2c_replay_gpu.answer_verdict``; it calls the
    same frozen ``contains_value``, so a value scored correct here is correct
    by the race's own rule.
    """
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


def a2_rescued(arms: Mapping[str, bool]) -> bool:
    """Did the order's A2 rescue, under EITHER sub-arm?

    A2 splits into A2a (the literal union, which this battery's arena width
    cannot seat) and A2b (the live-id node mounted alone).  "A2 rescues" means
    a MOUNT of that node text served correctly, and either assembly is such a
    mount, so either counts.  A bare ``"A2"`` key is still honoured so a
    caller holding a single-arm view is not silently read as "did not rescue".
    """
    return bool(
        arms.get("A2") or any(arms.get(name) for name in A2_ARMS))


def mechanism_verdict(arms: Mapping[str, bool]) -> str:
    """The registered vocabulary, applied by the registered rule.

    ``arms`` maps arm name -> "this arm served CORRECTLY".  The rule is the
    one written into the registration before any gate ran:

      QUANTIZATION if A3 rescues; else PRIOR if A4 rescues; else CONTENT if A2
      rescues; else POSITION if A1 rescues and A2 does not; else UNRESOLVED.

    ``A2`` is read through ``a2_rescued`` so the width-forced sub-arm split
    cannot change what the registered rule decides.

    A probe whose A0 was already correct is not a refuser and has no mechanism
    verdict to give; callers check that before calling.
    """
    if arms.get("A3"):
        return "QUANTIZATION"
    if arms.get("A4"):
        return "PRIOR"
    if a2_rescued(arms):
        return "CONTENT"
    if arms.get("A1"):
        return "POSITION"
    return "UNRESOLVED"


def assemble_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The G3 table: one row per (probe, arm), in probe-then-arm order.

    Pure shaping over already-measured rows, so the CPU tests can pin the
    column set without a GPU.
    """
    order = {arm: index for index, arm in enumerate(ARMS)}
    out: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda r: (str(r["probe_id"]), order.get(str(r["arm"]), 99)),
    ):
        mass = dict(row.get("mass") or {})
        mean = dict(mass.get("mean_over_answer_positions") or {})
        top = dict(mass.get("top_contributing_layer") or {})
        out.append({
            "probe_id": str(row["probe_id"]),
            "arm": str(row["arm"]),
            "served": str(row["served_answer"]),
            "correct": bool(row["correct"]),
            "mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
            "mounted_band_mass": mean.get("mounted_mass"),
            "live_band_mass": mean.get("live_mass"),
            "sink_mass": mean.get("physical_sink_mass"),
            "learned_sink_mass": mean.get("learned_sink_mass"),
            "top_contributing_layer": top.get("full_layer_ordinal"),
            "top_layer_mounted_mass": top.get("mounted_mass"),
        })
    return out


# ======================================================================
# The build
# ======================================================================
def _load_lived_repo(repo_dir: Path, frame: Mapping[str, Any], *,
                     storage_bits: int | None | str = "lived"):
    """The lived build, with ONE parameter this order needs to vary.

    ``storage_bits="lived"`` (the default) delegates straight to the read-only
    ``grm_det1_2_gpu._load_model_repo``, so arms A0/A1/A2/A4 get the exact
    lived construction and this module adds nothing to it.

    A3 needs ``storage_bits=None`` (the fp16 node payload).  ``_load_model_repo``
    hard-codes ``storage_bits=8`` and is READ-ONLY under this order's file
    boundary, so A3 constructs the repository here — from the FROZEN runtime
    frame's ``resolved_flags``, the same source and for the same reason as
    ``grm_lsr_p2c_1_gpu._open_probe_repo``.  Divergence from the lived build
    would silently change the arena the arm is measured on, so the flag values
    come off the frame rather than being retyped, and the ONE varied argument
    is named right where it is passed.
    """
    from scripts import grm_det1_2_gpu as det1_2

    if storage_bits == "lived":
        return det1_2._load_model_repo(repo_dir, frame)

    from transformers import AutoTokenizer

    from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e

    flags = frame["resolved_flags"]
    model, model_info = GptOss20B_TC.from_pretrained(det1_2.MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        str(det1_2.MODEL_DIR), local_files_only=True)
    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
        str(repo_dir / "repository"),
        autosave=False,
        arena_cls=e2e.GptOssGQAArenaCache,
        native_lib_path=str(det1_2.NATIVE_LIB),
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
        # THE ONE VARIED ARGUMENT: the lived build passes
        # flags["graft_storage_bits"] (== 8); A3 passes None, which is the
        # unquantized fp16 node payload.
        storage_bits=storage_bits,
        revision_resolution=bool(flags["sup_resolve"]),
        decisive_admission=bool(flags["adm_decisive"]),
    )
    return e2e, model, tokenizer, repo, model_info


def _probe_plan(session_id: str) -> list[dict[str, Any]]:
    """Every probe the LIVED ``_run_sup`` served for one fixture, IN ORDER.

    Delegates to ``lsr_p2c_replay_gpu.sup_probe_plan``: the probe ORDER is a
    property of the lived campaign, and reproducing EB1's rows requires the
    same order, so the frozen builder is imported rather than re-derived.
    """
    from scripts.lsr_p2c_replay_gpu import sup_probe_plan

    return sup_probe_plan(session_id)


# ======================================================================
# The serves
# ======================================================================
def _serve_ladder(repo, e2e, question: str, flags: Mapping[str, Any],
                  ) -> tuple[str, dict[str, Any], list[int], dict[str, Any]]:
    """A0/A3/A4: the PRODUCTION probe path, observed.

    ``e2e._probe_ladder_chat`` is the site the lived probes went through and
    the site P2A/P2C changed.  ``defer_memory=True`` matches the lived
    ``canonical_defer_memory=True``.  Nothing about the call changes when the
    observer is attached — the observer wraps ``arena._forward`` and the two
    attention kernels, reads, and puts them back.
    """
    from core import grm_demand

    arena = repo.arena
    with LayerMassObserver(
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
    """A1/A2: ONE generation attempt over an EXPLICIT mount set.

    ``arena._attempt`` is the call the ladder makes at its bottom; these two
    arms call it directly because their definition is "routing disabled" — a
    routed rung would re-decide the mount set, which is the one thing these
    arms fix by hand.  Everything else (the prompt format, the greedy decode,
    the stop sequences, ``defer_memory``) is the production call's own.

    THE WIDTH LAW STILL APPLIES, and it is applied through PRODUCTION's own
    packer.  ``_attempt``'s BOOTSTRAP branch (``self.caches is None``) injects
    every pick unconditionally — it has no width check, because in production
    nothing ever reaches it with an unfitted set: the ladder has already run
    ``_budget_fit_mounts``.  Measured once here, on the harbor A2 request
    {1, 2, 3}: the long ``opal_competitor`` node pushed the mount past
    ``arena_width`` (resident 266 seats against a width of 96) and the attempt
    served an EMPTY string.  That is a harness artifact, not a read-strength
    result, so this arm runs the same ``e2e._budget_fit_mounts`` the ladder
    runs and REPORTS what the fit dropped.  An arm whose requested set does not
    survive the fit is not silently downgraded — ``rs1_fit_dropped`` names the
    casualties on the receipt.
    """
    from core import grm_demand

    arena = repo.arena
    requested = [int(v) for v in picks]
    fitted = [int(v) for v in e2e._budget_fit_mounts(arena, requested)]
    dropped = [v for v in requested if v not in set(fitted)]
    with LayerMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
    ) as observer:
        answer, info = arena._attempt(
            question, fitted, int(flags["ngen"]), False,
            arena.stop_sequences or (), defer_memory=True)
    mounts = [int(v) for v in arena.cur_mounts]
    mount_n = int(arena.cur_mount_n)
    live_at_serve = [
        {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
        for g, n in arena.live_segs]
    grounded, _c = arena._grounding_attribution(str(answer), mounts, question)
    out = dict(info or {})
    out["rs1_grounded"] = bool(grounded)
    out["rs1_serving_path"] = "ArenaCache._attempt (routing disabled)"
    out["rs1_fit_requested"] = requested
    out["rs1_fit_seated"] = fitted
    out["rs1_fit_dropped"] = dropped
    out["rs1_arena_width"] = int(arena.width)
    out["rs1_cur_mount_n_at_serve"] = mount_n
    out["rs1_live_segs_at_serve"] = live_at_serve
    return str(answer), out, mounts, observer.partition()


def _feed_live(repo, e2e, fixture: Mapping[str, Any],
               node_ids: Sequence[str]) -> dict[str, Any]:
    """A1's seam: push node TEXT through the LIVE cache.

    Under the spec frame ``ArenaCache.feed`` deposits instead of pushing (see
    its own ``if self.ephemeral`` branch), so there is no live band to read
    from.  This pins ``ephemeral`` False for the duration of the feed — the
    same state ``GRM_PERSISTENT_BOAT`` selects, and the state every
    persistent-frame receipt in this repository was taken under — and restores
    it immediately after.  ``deposit=False`` so the feed adds NO graft: the arm
    is "the text is live and nothing is mounted", and a deposit would make the
    text routable, which is the opposite of the arm.
    """
    from scripts.grm_det1_3_gpu import _split_fixture_turn

    arena = repo.arena
    texts = {str(node["node_id"]): str(node["text"])
             for node in fixture["nodes"]}
    before = bool(getattr(arena, "ephemeral", False))
    grafts_before = len(arena.grafts)
    fed: list[dict[str, Any]] = []
    arena.ephemeral = False
    try:
        for node_id in node_ids:
            if str(node_id) not in texts:
                raise RS1Error(
                    f"registered live node {node_id!r} is not in fixture "
                    f"{fixture.get('session_id')!r}")
            user, assistant = _split_fixture_turn(texts[str(node_id)])
            turn_text = e2e.harmony_turn(user, assistant)
            ntok = len(arena.encode(turn_text))
            arena.feed(turn_text, deposit=False)
            fed.append({"node_id": str(node_id), "ntok": int(ntok)})
    finally:
        arena.ephemeral = before
    if len(arena.grafts) != grafts_before:
        raise RS1Error(
            "the A1 live feed deposited a graft; it must not — the arm is "
            f"live text with NOTHING mounted (before={grafts_before} "
            f"after={len(arena.grafts)})")
    return {
        "fed": fed,
        "live_segs_after_feed": [
            {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
            for g, n in arena.live_segs],
        "live_tokens_after_feed": int(sum(n for _, n in arena.live_segs)),
        "ephemeral_pinned_false_for_feed_only": True,
        "ephemeral_restored_to": bool(getattr(arena, "ephemeral", False)),
        "deposited_nothing": True,
    }


# ======================================================================
# The arm runner
# ======================================================================
def run_arm(session_id: str, arm: str) -> dict[str, Any]:
    """One arm over ONE fixture: build, then serve the fixture's probe plan."""
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual)

    if arm not in ARMS:
        raise RS1Error(f"unknown arm {arm!r}; registered arms are {ARMS}")
    registration = read_registration()
    targets = set(probes_for_session(registration, session_id))
    if not targets:
        raise RS1Error(f"no registered probe belongs to fixture {session_id}")
    probe_rows = registered_probes(registration)

    frame = json.loads(RUNTIME_FRAME.read_text(encoding="utf-8"))
    flags = frame["resolved_flags"]
    fixture = json.loads(
        (SUP_FIXTURES / f"{session_id}.json").read_text(encoding="utf-8"))
    plan = _probe_plan(session_id)

    # The lived switches, pinned for this process. Fixes ON (production),
    # demand OFF (the order: "fixes ON, demand OFF"), spec frame (the escape
    # is left UNSET so the fail-closed default selects the ephemeral boat).
    os.environ["GRM_LSR_FIXES"] = "1"
    os.environ["GRM_DEMAND_NGH"] = "0"
    os.environ.pop(PERSISTENT_BOAT_ENV, None)

    repo_dir = Path(tempfile.mkdtemp(prefix=f"grm_rs1_{session_id}_{arm}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    model_info: Any = None
    node_to_idx: dict[str, int] = {}
    live_at_install: list[int] = []
    frame_ephemeral = True
    storage_bits_observed: Any = None
    try:
        e2e, model, tokenizer, repo, model_info = _load_lived_repo(
            repo_dir, frame,
            storage_bits=(None if arm == "A3" else "lived"))
        arena = repo.arena
        frame_ephemeral = bool(getattr(arena, "ephemeral", False))
        storage_bits_observed = getattr(arena, "storage_bits", None)
        if arm == "A3" and storage_bits_observed is not None:
            raise RS1Error(
                "A3 asked for the fp16 node payload but the arena reports "
                f"storage_bits={storage_bits_observed!r}")
        if arm != "A3" and storage_bits_observed != int(
                flags["graft_storage_bits"]):
            raise RS1Error(
                f"{arm} must run on the LIVED storage payload "
                f"({flags['graft_storage_bits']} bits) but the arena reports "
                f"storage_bits={storage_bits_observed!r}")
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        live_at_install = [
            int(g) for g, _n in arena.live_segs if g is not None]

        for probe in plan:
            probe_id = str(probe["probe_id"])
            question = str(probe["question"])
            started = time.time_ns()
            feed_receipt: dict[str, Any] | None = None
            asked = question
            picks_requested: list[int] | None = None
            row = probe_rows.get(probe_id)
            # The mount seats THIS probe actually served on. A1/A2 restore the
            # arena after their serve, so reading ``arena.cur_mount_n`` after
            # that restore would report the state the NEXT probe inherits and
            # label it this probe's mount. Each branch sets this from its own
            # serve instead.
            mount_n_at_serve = 0

            if arm in ("A0", "A3"):
                answer, info, mounts, mass = _serve_ladder(
                    repo, e2e, question, flags)
                mount_n_at_serve = int(arena.cur_mount_n)
            elif arm == "A4":
                # A4 ASKS ABOUT THE GENERATION STAGE, so it must reach it.
                #
                # The first two A4 runs put the strict wording through the
                # WHOLE probe path, and both abstained before any mount work
                # (receipts kept). ``_probe_identifier_tokens`` reads the added
                # instruction words as IDENTIFIERS; nothing in the repository
                # binds "return"/"exactly"/"stored"/"sentence"/"punctuation"
                # (or the turn label's "50"), so P2A's identifier_unbound rule
                # fires and the turn serves "Not in memory: ..." with an EMPTY
                # mount and no mass rows at all. That measures the ABSTENTION
                # RULE, not the instruct prior — and the order's A4 is
                # explicitly about whether "the model READ the graft", which
                # requires the graft to be mounted.
                #
                # So A4 changes ONLY the stage it is about. Routing, admission
                # and fit run on the ORIGINAL question, giving A0's mount set;
                # generation then runs on the STRICT wording over exactly that
                # mount. The comparison against A0 is therefore one variable:
                # the words the model was asked to answer with.
                asked = strict_wording(question)
                route_answer, route_info, route_mounts, _rm = _serve_ladder(
                    repo, e2e, question, flags)
                # The routing serve IS A0's serve, and it must be the state
                # the NEXT probe inherits — so it is snapshotted here and
                # restored after the generation serve, exactly as A1/A2 do.
                base = _snapshot_counterfactual(arena)
                # Clear the boat so the generation serve starts from
                # [sink | mounts | this question], which is what the spec
                # frame's own turn-open leaves.
                arena.caches, arena.pos, arena.live_segs = None, 0, []
                arena.cur_mounts, arena.cur_mount_n = [], 0
                for layer in arena.m.layers:
                    layer.self_attn.live_shift = arena.live_shift
                answer, info, mounts, mass = _serve_direct(
                    repo, e2e, asked, flags, route_mounts)
                mount_n_at_serve = int(info["rs1_cur_mount_n_at_serve"])
                picks_requested = [int(v) for v in route_mounts]
                info["rs1_a4_routing_stage"] = {
                    "why": (
                        "routing/admission/fit ran on the ORIGINAL question "
                        "so the mount set is A0's; only the GENERATION "
                        "wording differs, which is the stage A4 is about"),
                    "routed_question": question,
                    "routed_served_answer": str(route_answer),
                    "routed_mounted_ids": [int(v) for v in route_mounts],
                    "routed_abstained": route_info.get("abstained"),
                }
                _restore_counterfactual(arena, base)
            elif row is None:
                # A1/A2 are DEFINED on the registered live ids, which exist
                # only for registered probes. A non-registered probe in the
                # fixture's plan is still SERVED — the state evolution the
                # registered probes inherit has to match A0's — but through
                # the A0 path, and the receipt says so.
                answer, info, mounts, mass = _serve_ladder(
                    repo, e2e, question, flags)
                info["rs1_arm_note"] = (
                    "not a registered probe: served through the A0 path so "
                    "the repository state this arm's registered probes "
                    "inherit matches A0's")
                mount_n_at_serve = int(arena.cur_mount_n)
            else:
                # THE A0 PRELUDE, and why the direct arms cannot skip it.
                # Two of the registered ``a0_mounted_ids`` are not stored
                # nodes at all: on ``fresh_fact_controls`` the long
                # ``sable_competitor`` (159 tokens against a width of 96) is
                # UNSEATABLE, so P2C's fit-time descent splits it and the
                # probe serves from child 4 — a graft that does not exist
                # until a fit has run. Measured: requesting id 4 on a freshly
                # installed repository raises IndexError out of
                # ``_budget_fit_mounts``, because ``arena.grafts`` holds three
                # nodes. The split is PERSISTED (``fit_split_ephemeral`` is
                # False; the librarian owns it), so running A0's own ladder
                # first materialises exactly the repository A0 had at this
                # probe. The direct arm then rebuilds only the BOAT on top of
                # it. This also keeps A1/A2's state evolution identical to
                # A0's, which is what makes probe N of a fixture comparable
                # across arms at all.
                prelude_answer, prelude_info, _pm, _pmass = _serve_ladder(
                    repo, e2e, question, flags)
                base = _snapshot_counterfactual(arena)
                missing = [
                    int(v) for v in row["a0_mounted_ids"]
                    if int(v) >= len(arena.grafts)]
                if missing:
                    raise RS1Error(
                        f"{probe_id}: registered a0_mounted_ids {missing} do "
                        f"not exist after the A0 prelude "
                        f"({len(arena.grafts)} grafts); the arm cannot mount "
                        "an id the repository does not hold")
                # Both arms start from an EMPTY boat, which is what the spec
                # frame's own turn-open produces; the two arms then differ
                # ONLY in where the payload is put.
                arena.caches, arena.pos, arena.live_segs = None, 0, []
                arena.cur_mounts, arena.cur_mount_n = [], 0
                for layer in arena.m.layers:
                    layer.self_attn.live_shift = arena.live_shift
                if arm == "A1":
                    feed_receipt = _feed_live(
                        repo, e2e, fixture,
                        [str(v) for v in row["live_node_ids"]])
                    picks_requested = []
                elif arm == "A2a":
                    # The LITERAL arm: A0's mount UNION the live ids.
                    picks_requested = sorted(
                        {int(v) for v in row["a0_mounted_ids"]}
                        | {int(v) for v in row["live_ids"]})
                else:
                    # A2b: A1's payload, moved from the live band to the
                    # mount band. A0's mount is NOT co-seated — that is the
                    # point: the only thing that changes between A1 and A2b is
                    # WHICH BAND the same node text sits in.
                    picks_requested = sorted(
                        int(v) for v in row["live_ids"])
                answer, info, mounts, mass = _serve_direct(
                    repo, e2e, question, flags, picks_requested)
                mount_n_at_serve = int(info["rs1_cur_mount_n_at_serve"])
                info["rs1_a0_prelude"] = {
                    "why": (
                        "A0's own ladder was run first so this arm's "
                        "repository is the one A0 had at this probe — "
                        "including any PERSISTED fit-time split whose child "
                        "the registered a0_mounted_ids name"),
                    "served_answer": str(prelude_answer),
                    "mounted_ids": [
                        int(v) for v in prelude_info.get("mount_fitted", ())],
                    "fit_split_parent": prelude_info.get("fit_split_parent"),
                    "fit_split_children": prelude_info.get(
                        "fit_split_children"),
                    "fit_split_ephemeral": prelude_info.get(
                        "fit_split_ephemeral"),
                    "grafts_after_prelude": len(arena.grafts),
                }
                # Leave the arena where the arm found it, so the NEXT probe in
                # the plan inherits A0's state evolution rather than this
                # arm's hand-built one.
                _restore_counterfactual(arena, base)

            verdict = answer_verdict(
                answer,
                expected_values=probe["expected_values"],
                rejected_values=probe["rejected_values"])
            served.append({
                "probe_id": probe_id,
                "session_id": session_id,
                "arm": arm,
                "registered_probe": probe_id in targets,
                "question": question,
                "question_asked": asked,
                "expected_values": list(probe["expected_values"]),
                "rejected_values": list(probe["rejected_values"]),
                "served_answer": str(answer),
                "correct": bool(verdict["correct"]),
                "verdict": verdict,
                "mounted_ids": [int(v) for v in mounts],
                "picks_requested": picks_requested,
                "live_feed": feed_receipt,
                "mass": mass,
                "info": {
                    key: value for key, value in dict(info).items()
                    if not str(key).startswith("_")
                },
                "eb1_lived_answer": str(probe["lived_answer"]),
                "arena_cur_mount_n": int(mount_n_at_serve),
                "elapsed_ns": int(time.time_ns() - started),
            })
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
        "phase": "RS1",
        "order": "orders/GRM_RS1_READ_STRENGTH_PROBE.md",
        "arm": arm,
        "session_id": session_id,
        "registered_probe_ids": sorted(targets),
        "frame": {
            "ephemeral": bool(frame_ephemeral),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
            "spec": (
                "GRM-EB1: the chat log is not kept in memory context; any "
                "chat recall on facts is pulled via GRM"),
        },
        "switches": {
            "GRM_LSR_FIXES": os.environ.get("GRM_LSR_FIXES"),
            "GRM_DEMAND_NGH": os.environ.get("GRM_DEMAND_NGH"),
            "storage_bits": storage_bits_observed,
        },
        "arena_width": int(flags["arena_width"]),
        "deposit_protocol": (
            "CHRONOLOGICAL_ARENA_FEED (grm_det1_3_gpu._install_lived_nodes), "
            "the LIVED arm"),
        "fixture_node_to_idx": {
            str(k): int(v) for k, v in node_to_idx.items()},
        "live_graft_ids_after_install": live_at_install,
        "probes": served,
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_rs1_read_strength_gpu": file_record(Path(__file__).resolve()),
            "grm_rs1_registration": file_record(
                ROOT / "scripts" / "grm_rs1_registration.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "lsr_p2c_replay_gpu": file_record(
                ROOT / "scripts" / "lsr_p2c_replay_gpu.py"),
        },
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
        raise RS1Error("--session-id is required for the arm command")
    if not args.arm:
        raise RS1Error("--arm is required for the arm command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_arm(str(args.session_id), str(args.arm))
    path = emit(payload, f"grm_rs1_{args.arm}_{args.session_id}")
    print(f"receipt={path}")
    for row in payload["probes"]:
        mean = dict(
            (row.get("mass") or {}).get("mean_over_answer_positions") or {})
        print(json.dumps({
            "arm": row["arm"],
            "probe_id": row["probe_id"],
            "registered": row["registered_probe"],
            "correct": row["correct"],
            "served": row["served_answer"][:90],
            "mounted_ids": row["mounted_ids"],
            "mounted_mass": mean.get("mounted_mass"),
            "live_mass": mean.get("live_mass"),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
