#!/usr/bin/env python3
"""GRM-RS4 — re-partition the live ceiling: fed text vs question vs answer.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md``.
REGISTRATION: ``artifacts/grm_rs4/registration.json`` (written FIRST, by
``scripts/grm_rs4_registration.py``; this module refuses to run without it and
never rewrites it).

WHY A RE-RUN WAS NECESSARY, stated plainly because the order made it
conditional.  Mission item 3 says "if per-key-row attention is not persisted in
the RS3 receipts, re-run".  It is not.  ``core/grm_demand.py::_full_mass``
softmaxes over the key rows and immediately SUMS them into four scalars per
layer; ``LayerTypeMassObserver`` keeps those four scalars and the four sliding
equivalents.  No RS3 artifact carries the per-row probabilities, so the live
band cannot be re-cut from disk at any granularity.  Hence this module, and
hence gate G2.

WHAT IS DIFFERENT FROM RS3, and it is the ONLY difference.  RS3 served each arm
under ``LayerTypeMassObserver``.  RS4 serves the SAME arms, through the SAME
production paths, under ``RowSplitMassObserver`` — a SUBCLASS that layers one
more recording wrapper on top and changes no arithmetic the base class
performs.  Concretely:

  * the base class's rows are still produced by the production
    ``DemandObserver``, still through its own ``_full_mass``, still checked by
    its own partition law.  RS4 never touches them.  That is what lets gate G2
    be a bit-equality gate rather than an approximate one;
  * on top of that, RS4 wraps ``gpt.sink_attention_tc`` ONE MORE TIME (the
    demand observer's own wrapper is already installed when RS4's ``__enter__``
    runs, so RS4's wrapper delegates to it and the production rows are
    untouched), recomputes the same softmax with the same operands, and cuts
    the LIVE band into the four registered sub-bands;
  * the same is done for the sliding layers, intersected with the window slice
    the sliding kernel actually scores, exactly as RS2 registered.

MEASUREMENT ONLY.  Nothing under ``core/`` changes.  No flag default moves.
The arms are RS3's arms at RS3's registered lever values, read off RS3's
registration rather than re-typed.

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

import numpy as np

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
from scripts.grm_rs3_capture_seat_gpu import (  # noqa: E402
    CAPTURE_ARMS, _recapture_under_pin,
)
from scripts.grm_rs3_capture_seat_gpu import (  # noqa: E402
    REGISTRATION as RS3_REGISTRATION_PATH,
)
from scripts.grm_rs3_capture_seat_gpu import (  # noqa: E402
    read_registration as read_rs3_registration,
)
from scripts.grm_rs4_registration import ARMS, LIVE_SUBBANDS, RS4Error  # noqa: E402
from scripts.grm_rs4_row_split import (  # noqa: E402
    ROW_NAMES, live_band_bounds, mean_over_positions, partition_sums_to_one,
    split_full_row, subbands_sum_to_live,
)

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs4"
REGISTRATION = ARTIFACT_DIR / "registration.json"
SCHEMA_PREFIX = "grm.rs4"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 560
LOCK_WAIT_SECONDS = 7200


# ======================================================================
# Registration reading (pure — CPU tested)
# ======================================================================
def read_registration(path: Path = REGISTRATION) -> dict[str, Any]:
    if not Path(path).is_file():
        raise RS4Error(
            f"registration missing: {path}. Write it FIRST with "
            "scripts/grm_rs4_registration.py; no gate may run without it.")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arm_flags(registration: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """This arm's lever values, READ OFF RS3's registration, not RS4's.

    RS4 re-runs RS3's arms.  Taking the levers from RS3's own registration —
    the file the RS3 receipts were measured against — is what makes "the same
    arm" a checkable claim rather than a transcription.  RS4's own registration
    carries a human-readable label for each arm and is cross-checked here.
    """
    if arm not in ARMS:
        raise RS4Error(f"unknown arm {arm!r}; RS4 arms are {ARMS}")
    rs3 = read_rs3_registration()["arms_REGISTERED_BEFORE_ANY_GATE"]
    if arm not in rs3:
        raise RS4Error(
            f"arm {arm!r} is not in RS3's registration; RS4 only re-runs arms "
            "RS3 registered")
    entry = rs3[arm]
    mine = registration["arms_REGISTERED_BEFORE_ANY_GATE"][arm]
    if (str(entry["capture_pin"]) != str(mine["capture_pin"])
            or bool(entry["seat_near_live"]) != bool(mine["seat_near_live"])):
        raise RS4Error(
            f"arm {arm!r} levers disagree between RS3 and RS4 registrations: "
            f"RS3 says pin={entry['capture_pin']} "
            f"seat={entry['seat_near_live']}, RS4 says "
            f"pin={mine['capture_pin']} seat={mine['seat_near_live']}")
    return {
        "capture_pin": str(entry["capture_pin"]),
        "seat_near_live": bool(entry["seat_near_live"]),
        "label": str(entry["label"]),
        "levers_read_from": "artifacts/grm_rs3/registration.json",
    }


def registered_probes(registration: Mapping[str, Any]) -> dict[str, Any]:
    """RS3's probe block — the probes, mount ids and live node ids RS4 reuses."""
    return dict(read_rs3_registration()[
        "probes_REGISTERED_BEFORE_ANY_GATE"])


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


def solace_variant_applies(registration: Mapping[str, Any],
                           probe_id: str) -> bool:
    variant = read_rs3_registration().get(
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE") or {}
    return str(variant.get("probe_id", "")) == str(probe_id)


def arm_plan(registration: Mapping[str, Any],
             session_id: str) -> list[dict[str, Any]]:
    """One work item per (probe, arm, variant) for this fixture.

    RS3's plan, restricted to RS4's four arms.  The solace probe still gets the
    ``solace_fact`` variant of every arm, for RS2's measured reason (RS1's A0
    mounted a FIT-TIME SPLIT CHILD there, not the solace fact), because the
    residual this order measures would otherwise be confounded on that row by
    the wrong-graft finding rather than by anything about bands.
    """
    rs3 = read_rs3_registration()
    probes = dict(rs3["probes_REGISTERED_BEFORE_ANY_GATE"])
    variant_spec = rs3.get(
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


# ======================================================================
# The extended instrument — RS4's ONLY new measurement
# ======================================================================
class RowSplitMassObserver(LayerTypeMassObserver):
    """RS2/RS3's instrument, with the LIVE band cut into its four sub-bands.

    THE BASE CLASS IS NOT MODIFIED AND ITS ROWS ARE NOT TOUCHED.  Everything
    RS3 reported still comes out of ``LayerTypeMassObserver.partition()``,
    computed by the production ``DemandObserver`` for the full layers and by
    RS2's registered sliding partition for the sliding ones.  That is what
    makes gate G2 a BIT-EQUALITY gate: if RS4's addition had perturbed the base
    computation at all, C0 could not reproduce RS3 exactly, and it does.

    WHAT THIS CLASS ADDS.  One more wrapper on ``gpt.sink_attention_tc`` and
    one more on ``gpt.sliding_sink_attention_tc``, installed AFTER the base
    class has installed its own (so RS4's delegate to theirs and the ordering
    is: RS4 records -> RS2 records -> production records -> real kernel).  Each
    recomputes the softmax from the same operands the production path used and
    sums it into the registered sub-bands.

    THE ROW BOUNDARIES ARE NOT PASSED IN AS NUMBERS.  ``fed_ntok`` and
    ``question_ntok`` are handed to the constructor because they are properties
    of the SERVE (what this arm fed, what question it is about to ask), and
    both are computed by the caller from the arm's own feed receipt and the
    model's own tokenizer.  ``prior_ntok`` is read off ``arena.pos`` at
    ``__enter__``.  ``S``, ``n_sink`` and ``cur_mount_n`` are read off the live
    kernel call and the live arena, per forward.  No row index is a literal.
    """

    def __init__(self, arena: Any, ngen: int, threshold: float, *,
                 fed_ntok: int, question_ntok: int) -> None:
        super().__init__(arena, int(ngen), float(threshold))
        self.fed_ntok = int(fed_ntok)
        self.question_ntok = int(question_ntok)
        self.pos_at_entry: int | None = None
        self.prior_ntok: int | None = None
        self.split_full_rows: list[list[dict[str, Any]]] = []
        self.split_sliding_rows: list[list[dict[str, Any]]] = []
        self._pending_full_split: list[dict[str, Any]] = []
        self._pending_sliding_split: list[dict[str, Any]] = []
        self._rs4_original_sink = None
        self._rs4_original_sliding = None
        self._rs4_original_forward = None
        self.bounds_seen: list[dict[str, Any]] = []
        self._pos_before_forward: int | None = None
        self._ids_this_forward: int | None = None
        self.baseline_ntok: int | None = None
        self.baseline_receipt: dict[str, Any] | None = None

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "RowSplitMassObserver":
        super().__enter__()
        # The base class (and, under it, the production observer) have now
        # installed their wrappers.  Capture THOSE as the delegates, so RS4
        # sits strictly outside and perturbs neither.
        gpt = self._gpt
        observer = self
        self._rs4_original_sink = gpt.sink_attention_tc
        self._rs4_original_sliding = gpt.sliding_sink_attention_tc
        self._rs4_original_forward = self.arena._forward

        # ``arena.pos`` at ENTRY is recorded for the receipt but it is NOT the
        # baseline; see ``_baseline_from_first_forward`` for the MEASURED
        # reason and the derivation that replaced it.
        self.pos_at_entry = int(getattr(self.arena, "pos", 0) or 0)

        def sink_wrapper(query, key, value, sinks, *, scale,
                         attention_mask=None, num_heads_per_kv=1):
            result = observer._rs4_original_sink(
                query, key, value, sinks,
                scale=scale,
                attention_mask=attention_mask,
                num_heads_per_kv=int(num_heads_per_kv),
            )
            inner = observer._inner
            if inner._inside_forward and not inner._suppress_sink:
                observer._pending_full_split.append(observer._full_split(
                    query, key, sinks,
                    scale=float(scale),
                    attention_mask=attention_mask,
                    num_heads_per_kv=int(num_heads_per_kv),
                ))
            return result

        def sliding_wrapper(query, key, value, sinks, *, scale,
                            sliding_window, num_heads_per_kv=1, attn_block=128,
                            allowed_mask=None):
            result = observer._rs4_original_sliding(
                query, key, value, sinks,
                scale=scale,
                sliding_window=sliding_window,
                num_heads_per_kv=int(num_heads_per_kv),
                attn_block=int(attn_block),
                allowed_mask=allowed_mask,
            )
            if observer._inner._inside_forward:
                observer._pending_sliding_split.append(
                    observer._sliding_split(
                        query, key, sinks,
                        scale=float(scale),
                        sliding_window=int(sliding_window),
                        num_heads_per_kv=int(num_heads_per_kv),
                        allowed_mask=allowed_mask,
                    ))
            return result

        def forward_wrapper(ids, last_only=True):
            observer._pending_full_split = []
            observer._pending_sliding_split = []
            # The live-row counter AS THE FORWARD BEGINS. This is the number
            # the baseline is derived from, and it is read here rather than at
            # __enter__ for the measured reason in
            # ``_baseline_from_first_forward``.
            observer._pos_before_forward = int(
                getattr(observer.arena, "pos", 0) or 0)
            observer._ids_this_forward = int(len(ids))
            try:
                return observer._rs4_original_forward(ids, last_only=last_only)
            finally:
                # Closed on the way out whether or not the inner forward
                # raised, so RS4's rows stay aligned with the base class's.
                observer.split_full_rows.append(
                    list(observer._pending_full_split))
                observer.split_sliding_rows.append(
                    list(observer._pending_sliding_split))
                observer._pending_full_split = []
                observer._pending_sliding_split = []

        gpt.sink_attention_tc = sink_wrapper
        gpt.sliding_sink_attention_tc = sliding_wrapper
        self.arena._forward = forward_wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._rs4_original_forward is not None:
            self.arena._forward = self._rs4_original_forward
        gpt = self._gpt
        if gpt is not None:
            if self._rs4_original_sink is not None:
                gpt.sink_attention_tc = self._rs4_original_sink
            if self._rs4_original_sliding is not None:
                gpt.sliding_sink_attention_tc = self._rs4_original_sliding
        return super().__exit__(exc_type, exc, tb)

    # -- the split ------------------------------------------------------
    def _baseline_from_first_forward(self, S: int) -> int:
        """Live rows already in the cache when this serve's PROMPT went in.

        MEASURED FINDING, and the reason this is not simply ``arena.pos`` at
        ``__enter__``.  Under the SPEC frame every production serving path
        opens its turn through ``ArenaCache.eb1_begin_turn``, whose ephemeral
        branch does ``self.caches, self.pos, self.live_segs = None, 0, []`` —
        it CLEARS THE BOAT.  ``_probe_ladder_chat`` calls it, so on the C0
        ladder the live rows the observer sees at entry are NOT in the cache
        the kernel then scores.  RS4's first C0 run raised on exactly that:
        ``live0=83 prior=48 fed=0 question=37 ends at 168 but S=120`` — 48
        phantom rows that the turn-open had already discarded.  The guard did
        its job; the derivation was wrong, and this is the corrected one.

        THE CORRECTED DERIVATION reads the state the kernel actually has.  The
        FIRST forward inside a serve is ``_attempt``'s prompt forward, which
        pushes ``question_ntok`` rows onto whatever the cache already held.  So

            baseline = S_first - live0 - question_ntok

        where every term on the right is observed: ``S_first`` is the key-row
        count the kernel was handed on that first call, ``live0`` is
        ``n_sink + cur_mount_n`` off the live arena, and ``question_ntok`` is
        the model tokenizer's count of ``_attempt``'s own prompt string.  It is
        cross-checked against ``arena.pos`` as the forward began — the same
        number by a different route — and a disagreement raises.

        CHECKED AGAINST RS3 BEFORE IT WAS USED: on RS3's C5 harbor receipt
        ``S_first = 266``, ``live0 = 19``, ``question_ntok = 36``, giving
        ``baseline = 211``, which is exactly the ``live_tokens_after_feed``
        that receipt records for the feed.
        """
        if self.baseline_ntok is not None:
            return int(self.baseline_ntok)
        live0 = int(self.arena.n_sink) + int(self.arena.cur_mount_n)
        derived = int(S) - live0 - int(self.question_ntok)
        observed_pos = self._pos_before_forward
        if derived < 0:
            raise RS4Error(
                f"the first observed forward scored S={S} rows with "
                f"live0={live0} and a {self.question_ntok}-token question: "
                "the question alone overruns the cache, so the prompt this "
                "serve encoded is not the one the registration describes")
        if observed_pos is not None and int(observed_pos) != derived:
            raise RS4Error(
                f"live-row baseline disagrees between its two derivations: "
                f"S - live0 - question_ntok = {derived} but arena.pos at the "
                f"forward was {observed_pos}. The row split would be cutting "
                "the live band at rows the kernel did not score.")
        self.baseline_ntok = int(derived)
        self.baseline_receipt = {
            "S_first_forward": int(S),
            "live0": int(live0),
            "question_ntok": int(self.question_ntok),
            "baseline_ntok_derived": int(derived),
            "arena_pos_before_first_forward": (
                None if observed_pos is None else int(observed_pos)),
            "two_derivations_agree": True,
            "fed_ntok_from_feed_receipt": int(self.fed_ntok),
            "prior_ntok": int(derived) - int(self.fed_ntok),
            "why_not_pos_at_observer_entry": (
                "ArenaCache.eb1_begin_turn's ephemeral branch clears the boat "
                "(caches, pos, live_segs = None, 0, []) at every production "
                "turn-open, so rows present at observer entry may not be in "
                "the cache the kernel scores. MEASURED on RS4's first C0 run."),
        }
        return int(self.baseline_ntok)

    def _bounds_for(self, S: int) -> dict[str, Any]:
        baseline = self._baseline_from_first_forward(int(S))
        prior = baseline - int(self.fed_ntok)
        if prior < 0:
            raise RS4Error(
                f"the live-row baseline at the prompt forward is {baseline} "
                f"but the arm's own feed receipt claims {self.fed_ntok} fed "
                "tokens; the feed and the cache disagree")
        self.prior_ntok = int(prior)
        bounds = live_band_bounds(
            S=int(S),
            n_sink=int(self.arena.n_sink),
            cur_mount_n=int(self.arena.cur_mount_n),
            prior_ntok=int(prior),
            fed_ntok=int(self.fed_ntok),
            question_ntok=int(self.question_ntok),
        )
        self.bounds_seen.append(dict(bounds))
        return bounds

    def _full_split(self, query, key, sinks, *, scale: float, attention_mask,
                    num_heads_per_kv: int) -> dict[str, Any]:
        """``_full_mass``'s computation, operand for operand, then re-cut.

        Deliberately the SAME operations in the SAME order as
        ``core/grm_demand.py::_full_mass``: ``_repeat_kv``, ``matmul`` with
        ``trans_b`` and ``alpha=scale``, the mask slice, the ``cat`` of the
        sink logit, ``softmax(-1)``.  Only the summation changes, and only by
        stopping the live sum at three interior row indices.  Any divergence
        from that line-up is exactly what gate G2's bit-identity check on the
        UNCHANGED bands would catch.
        """
        gpt = self._gpt
        q_rows = int(query.shape[2])
        key_rows = int(key.shape[2])
        q_last = query.slice(2, q_rows - 1, 1)
        key_h = gpt._repeat_kv(key, int(num_heads_per_kv))
        scores = gpt.tc.matmul(q_last, key_h, alpha=float(scale), trans_b=True)
        if attention_mask is not None:
            scores = scores + attention_mask.slice(2, q_rows - 1, 1)
        batch, heads, _one, _source = (int(scores.shape[i]) for i in range(4))
        sink_logits = sinks.reshape([1, heads, 1, 1]).expand(
            [batch, heads, 1, 1])
        probs = gpt.tc.cat([scores, sink_logits], dim=-1).softmax(-1)
        values = np.asarray(probs.float().numpy(), dtype=np.float32)[0, :, 0, :]

        bounds = self._bounds_for(key_rows)
        row = split_full_row(values, bounds)
        row["layer_type"] = FULL
        row["bounds"] = bounds
        row["subbands_sum_to_live"] = bool(subbands_sum_to_live(row))
        row["partition_sums_to_one"] = bool(partition_sums_to_one(row))
        return row

    def _sliding_split(self, query, key, sinks, *, scale: float,
                       sliding_window: int, num_heads_per_kv: int,
                       allowed_mask) -> dict[str, Any]:
        """RS2's registered sliding partition, re-cut into the sub-bands.

        The window slice and the mask are RS2's, unchanged: the sliding kernel
        only scores ``[k0, k1)`` for the last query row, so a sub-band outside
        the window contributes a STRUCTURAL zero, recorded as one alongside the
        intersection width rather than being reported as a measured near-zero.
        """
        gpt = self._gpt
        L = int(query.shape[2])
        S = int(key.shape[2])
        window = max(1, int(sliding_window))
        q_abs = S - 1
        k0 = max(0, q_abs - window + 1)
        k1 = min(S, q_abs + 1)

        q_last = query.slice(2, L - 1, 1)
        key_slice = key.slice(2, k0, k1 - k0)
        key_h = gpt._repeat_kv(key_slice, int(num_heads_per_kv))
        scores = gpt.tc.matmul(q_last, key_h, alpha=float(scale), trans_b=True)

        if allowed_mask is None:
            k_abs = np.arange(k0, k1, dtype=np.int64)
            allowed = (k_abs <= q_abs) & (k_abs > (q_abs - window))
        else:
            full_allowed = np.asarray(allowed_mask, dtype=np.uint8)
            if full_allowed.shape != (1, 1, L, S):
                raise RS4Error(
                    "sliding allowed-mask shape mismatch: "
                    f"expected={(1, 1, L, S)} observed={full_allowed.shape}")
            allowed = full_allowed[0, 0, L - 1, k0:k1].astype(bool, copy=False)
        mask = np.where(allowed, 0.0, -1.0e4).astype(np.float32)
        mask_t = gpt.tc.tensor(
            mask.reshape(1, 1, 1, k1 - k0), dtype="float32").astype(query.dtype)
        scores = scores + mask_t

        batch, heads, _one, _source = (int(scores.shape[i]) for i in range(4))
        sink_logits = sinks.reshape([1, heads, 1, 1]).expand(
            [batch, heads, 1, 1])
        probs = gpt.tc.cat([scores, sink_logits], dim=-1).softmax(-1)
        values = np.asarray(probs.float().numpy(), dtype=np.float32)[0, :, 0, :]

        bounds = self._bounds_for(S)
        scored = k1 - k0

        def band(name: str) -> tuple[float, int]:
            lo, hi = (int(v) for v in bounds[name])
            lo_c, hi_c = max(lo, k0), min(hi, k1)
            if hi_c <= lo_c:
                return 0.0, 0
            return (float(np.mean(
                values[:, lo_c - k0:hi_c - k0].sum(axis=1))), hi_c - lo_c)

        sink_mass, sink_rows = band("sink_band")
        mounted_mass, mount_rows = band("mount_band")
        live_mass, live_rows = band("live_band")
        prior_mass, prior_rows = band("prior_live_band")
        fed_mass, fed_rows = band("fed_text_band")
        question_mass, question_rows = band("question_band")
        answer_mass, answer_rows = band("answer_band")

        row: dict[str, Any] = {
            "layer_type": SLIDING,
            "physical_sink_mass": sink_mass,
            "mounted_mass": mounted_mass,
            "live_mass": live_mass,
            "learned_sink_mass": float(np.mean(values[:, scored])),
            "prior_live_mass": prior_mass,
            "fed_text_mass": fed_mass,
            "question_mass": question_mass,
            "answer_mass": answer_mass,
            "bounds": bounds,
            "k0": int(k0),
            "k1": int(k1),
            "sliding_window": int(window),
            "rows_scored": int(scored),
            "sink_rows_in_window": int(sink_rows),
            "mount_rows_in_window": int(mount_rows),
            "live_rows_in_window": int(live_rows),
            "prior_live_rows_in_window": int(prior_rows),
            "fed_text_rows_in_window": int(fed_rows),
            "question_rows_in_window": int(question_rows),
            "answer_rows_in_window": int(answer_rows),
            "window_reaches_mount": bool(mount_rows > 0),
            "window_reaches_fed_text": bool(fed_rows > 0),
            "fed_text_band_width": int(bounds["fed_text_rows"]),
            "fed_text_structurally_invisible": bool(
                int(bounds["fed_text_rows"]) > 0 and fed_rows == 0),
        }
        row["subbands_sum_to_live"] = bool(subbands_sum_to_live(row))
        return row

    # -- the table ------------------------------------------------------
    def split_partition(self) -> dict[str, Any]:
        """The RS4 table for ONE serve, beside the base class's own.

        ``base`` is ``LayerTypeMassObserver.partition()`` VERBATIM — the RS3
        numbers, so a receipt carries both framings and G2 can compare the old
        one without re-deriving it.  ``split`` is the new cut.
        """
        base = self.partition()
        n = int(base["answer_positions"])
        full_steps = self.split_full_rows[:n]
        sliding_steps = self.split_sliding_rows[:n]
        split = {
            FULL: mean_over_positions(full_steps, ROW_NAMES),
            SLIDING: mean_over_positions(sliding_steps, ROW_NAMES),
        }
        for name, block in split.items():
            block["layer_type"] = name

        # THE AGREEMENT CHECK.  The re-cut live band must equal the live band
        # the untouched instrument reports, for both layer types.  This is what
        # says the sub-bands are a re-partition of RS3's own number and not a
        # second computation that happens to be close.
        agreement: dict[str, Any] = {}
        for name in (FULL, SLIDING):
            base_mean = dict(
                dict(base["by_layer_type"][name])
                .get("mean_over_answer_positions") or {})
            new_mean = dict(split[name]["mean_over_answer_positions"])
            agreement[name] = {
                key: (
                    None if base_mean.get(key) is None
                    or new_mean.get(key) is None
                    else abs(float(base_mean[key]) - float(new_mean[key])))
                for key in ("physical_sink_mass", "mounted_mass", "live_mass",
                            "learned_sink_mass")
            }
        return {
            "answer_positions": n,
            "row_split_inputs": {
                "pos_at_observer_entry": self.pos_at_entry,
                "baseline_ntok": self.baseline_ntok,
                "baseline_receipt": self.baseline_receipt,
                "prior_ntok": self.prior_ntok,
                "fed_ntok": int(self.fed_ntok),
                "question_ntok": int(self.question_ntok),
                "derived_by": (
                    "baseline = S_first - (n_sink + cur_mount_n) - "
                    "question_ntok, cross-checked against arena.pos as the "
                    "first observed forward began; fed off the arm's feed "
                    "receipt; prior = baseline - fed; question through the "
                    "model tokenizer over arena._format_step_prompt; S, "
                    "n_sink and cur_mount_n read per forward off the live "
                    "call and the live arena"),
            },
            "bounds_first_seen": (
                dict(self.bounds_seen[0]) if self.bounds_seen else None),
            "bounds_last_seen": (
                dict(self.bounds_seen[-1]) if self.bounds_seen else None),
            "by_layer_type": split,
            "base_partition_UNMODIFIED": base,
            "recut_vs_base_abs_delta": agreement,
            "sliding_fed_text_reach": _fed_reach(sliding_steps),
        }


def _fed_reach(steps: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Whether the sliding window ever reached the FED-TEXT rows.

    RS2 registered the rule that a structural zero must be distinguishable from
    a measured near-zero.  The fed-text band is far below the query on C5 (it
    is the oldest live content), so on a 128-wide window it can fall entirely
    out of the slice; when it does, the sliding fed-text mass is a structural
    zero and must be READ as one.  Pure over captured rows.
    """
    flat = [dict(row) for step in steps for row in step]
    if not flat:
        return {"observed_rows": 0, "ever_reaches_fed_text": None}
    reaches = [bool(row["window_reaches_fed_text"]) for row in flat]
    widths = [int(row["fed_text_band_width"]) for row in flat]
    return {
        "observed_rows": len(flat),
        "ever_reaches_fed_text": bool(any(reaches)),
        "always_reaches_fed_text": bool(all(reaches)),
        "reaching_rows": int(sum(1 for value in reaches if value)),
        "fed_text_band_width_max": int(max(widths)),
        "structurally_invisible_rows": int(sum(
            1 for row in flat if row["fed_text_structurally_invisible"])),
    }


# ======================================================================
# The serves — RS3's paths, under RS4's instrument
# ======================================================================
def _question_ntok(arena, question: str) -> int:
    """``len(encode(_format_step_prompt(q)))`` — the arena's OWN two calls.

    Not a lookup in the registration: the arm asks the live arena what it is
    about to encode, so a template or tokenizer that had drifted would produce
    a receipt whose number disagrees with the registration and be caught, not
    a receipt that silently matched a stale expectation.
    """
    return int(len(arena.encode(arena._format_step_prompt(str(question)))))


def _serve_ladder(repo, e2e, question: str, flags: Mapping[str, Any]):
    """C0: the PRODUCTION probe path, observed with RS4's instrument."""
    from core import grm_demand

    arena = repo.arena
    with RowSplitMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
        fed_ntok=0, question_ntok=_question_ntok(arena, question),
    ) as observer:
        answer, info = e2e._probe_ladder_chat(
            repo, question,
            topk=int(flags["topk"]),
            ngen=int(flags["ngen"]),
            max_trips=int(flags["max_trips"]),
            defer_memory=True,
        )
    return (str(answer), dict(info or {}),
            [int(v) for v in arena.cur_mounts], observer.split_partition())


def _serve_direct(repo, e2e, question: str, flags: Mapping[str, Any],
                  picks: Sequence[int], fed_ntok: int):
    """Every non-C0 arm: ONE ``_attempt`` over an EXPLICIT mount set.

    RS3's ``_serve_direct``, unchanged except for the instrument and for
    carrying ``fed_ntok`` into it.  The serving path is still PRODUCTION's
    ``_attempt``, bootstrap seating branch included.
    """
    from core import grm_demand

    arena = repo.arena
    requested = [int(v) for v in picks]
    fitted = [int(v) for v in e2e._budget_fit_mounts(arena, requested)]
    dropped = [v for v in requested if v not in set(fitted)]
    with RowSplitMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
        fed_ntok=int(fed_ntok),
        question_ntok=_question_ntok(arena, question),
    ) as observer:
        answer, info = arena._attempt(
            question, fitted, int(flags["ngen"]), False,
            arena.stop_sequences or (), defer_memory=True)
    mounts = [int(v) for v in arena.cur_mounts]
    grounded, _c = arena._grounding_attribution(str(answer), mounts, question)
    out = dict(info or {})
    out["rs4_grounded"] = bool(grounded)
    out["rs4_serving_path"] = (
        "ArenaCache._attempt (routing disabled) — PRODUCTION, including the "
        "bootstrap seating branch the seat lever lives in")
    out["rs4_fit_requested"] = requested
    out["rs4_fit_seated"] = fitted
    out["rs4_fit_dropped"] = dropped
    out["rs4_arena_width"] = int(arena.width)
    out["rs4_cur_mount_n_at_serve"] = int(arena.cur_mount_n)
    out["rs4_seating"] = getattr(arena, "_rs3_last_seating", None)
    return str(answer), out, mounts, observer.split_partition()


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

    Carries BOTH framings side by side on every row: the RS3 columns
    (mounted / live / sink / learned sink, off the UNMODIFIED base partition,
    so an RS4 cell and an RS3 cell are the same number) and RS4's four live
    sub-bands.  Pure shaping over already-measured rows.
    """
    order = {arm: index for index, arm in enumerate(ARMS)}
    out: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda r: (str(r["probe_id"]), str(r.get("variant", "registered")),
                       order.get(str(r["arm"]), len(order))),
    ):
        mass = dict(row.get("mass") or {})
        base = dict(mass.get("base_partition_UNMODIFIED") or {})
        base_by_type = dict(base.get("by_layer_type") or {})
        base_full = dict(
            dict(base_by_type.get(FULL) or {})
            .get("mean_over_answer_positions") or {})
        base_sliding = dict(
            dict(base_by_type.get(SLIDING) or {})
            .get("mean_over_answer_positions") or {})
        split_by_type = dict(mass.get("by_layer_type") or {})
        split_full = dict(
            dict(split_by_type.get(FULL) or {})
            .get("mean_over_answer_positions") or {})
        split_sliding = dict(
            dict(split_by_type.get(SLIDING) or {})
            .get("mean_over_answer_positions") or {})
        inputs = dict(mass.get("row_split_inputs") or {})
        bounds = dict(mass.get("bounds_first_seen") or {})
        entry: dict[str, Any] = {
            "probe_id": str(row["probe_id"]),
            "variant": str(row.get("variant", "registered")),
            "arm": str(row["arm"]),
            "levers": dict(row.get("levers") or {}),
            "served": str(row["served_answer"]),
            "correct": bool(row["correct"]),
            "mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
            "answer_positions": mass.get("answer_positions"),
            # RS3's columns, off the UNMODIFIED base partition.
            "mounted_mass_full_layers": base_full.get("mounted_mass"),
            "live_mass_full_layers": base_full.get("live_mass"),
            "sink_mass_full_layers": base_full.get("physical_sink_mass"),
            "learned_sink_mass_full_layers": base_full.get(
                "learned_sink_mass"),
            "mounted_mass_sliding_layers": base_sliding.get("mounted_mass"),
            "live_mass_sliding_layers": base_sliding.get("live_mass"),
            # RS4's columns: the live band, re-cut.
            "row_split_inputs": inputs,
            "row_bounds": bounds,
            "subbands_sum_to_live_full": dict(
                split_by_type.get(FULL) or {}).get("subbands_sum_to_live"),
            "partition_sums_to_one_full": dict(
                split_by_type.get(FULL) or {}).get("partition_sums_to_one"),
            "sliding_fed_text_reach": mass.get("sliding_fed_text_reach"),
            "recut_vs_base_abs_delta": mass.get("recut_vs_base_abs_delta"),
        }
        for name in LIVE_SUBBANDS:
            entry[f"{name}_full_layers"] = split_full.get(name)
            entry[f"{name}_sliding_layers"] = split_sliding.get(name)
        out.append(entry)
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
    """One arm over ONE fixture: build, then serve the fixture's probe plan.

    STRUCTURALLY RS3's ``run_arm``, so that C0 can reproduce RS3 bit-equal:
    the same build, the same lived install, the same whole-plan replay in the
    lived order, the same prelude, the same counterfactual rollback.  The two
    differences are the instrument and the fact that only four arms exist here.
    """
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual)

    if arm not in ARMS:
        raise RS4Error(f"unknown arm {arm!r}; RS4 arms are {ARMS}")
    registration = read_registration()
    targets = set(probes_for_session(registration, session_id))
    if not targets:
        raise RS4Error(f"no registered probe belongs to fixture {session_id}")
    levers = arm_flags(registration, arm)
    plan_items = [item for item in arm_plan(registration, session_id)
                  if item["arm"] == arm]

    frame = json.loads(RUNTIME_FRAME.read_text(encoding="utf-8"))
    flags = frame["resolved_flags"]
    fixture = json.loads(
        (SUP_FIXTURES / f"{session_id}.json").read_text(encoding="utf-8"))
    plan = _probe_plan(session_id)

    os.environ["GRM_LSR_FIXES"] = "1"
    os.environ["GRM_DEMAND_NGH"] = "0"
    os.environ.pop(PERSISTENT_BOAT_ENV, None)
    # The two RS3 levers are passed EXPLICITLY, never through the env, and the
    # env is cleared so a leaked value cannot reach the OFF arms either.
    os.environ.pop(CAPTURE_PIN_ENV, None)
    os.environ.pop(SEAT_NEAR_LIVE_ENV, None)

    repo_dir = Path(tempfile.mkdtemp(prefix=f"grm_rs4_{session_id}_{arm}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    model_info: Any = None
    node_to_idx: dict[str, int] = {}
    live_at_install: list[int] = []
    frame_ephemeral = True
    storage_bits_observed: Any = None
    layer_inventory: dict[str, Any] = {}
    geometry_observed: dict[str, Any] = {}
    question_ntok_checks: list[dict[str, Any]] = []
    try:
        e2e, model, tokenizer, repo, model_info = _load_lived_repo(
            repo_dir, frame, storage_bits="lived")
        arena = repo.arena
        frame_ephemeral = bool(getattr(arena, "ephemeral", False))
        storage_bits_observed = getattr(arena, "storage_bits", None)
        if storage_bits_observed != int(flags["graft_storage_bits"]):
            raise RS4Error(
                f"{arm} must run on the LIVED storage payload "
                f"({flags['graft_storage_bits']} bits) but the arena reports "
                f"storage_bits={storage_bits_observed!r}")
        registered_geometry = registration[
            "geometry_REGISTERED_BEFORE_ANY_GATE"]
        geometry_observed = {
            "n_sink": int(arena.n_sink),
            "arena_width": int(arena.width),
            "live_shift": int(arena.live_shift),
        }
        for key, value in geometry_observed.items():
            if int(registered_geometry[key]) != value:
                raise RS4Error(
                    f"registered {key}={registered_geometry[key]} but the "
                    f"live arena reports {value}: the registration does not "
                    "describe the band being measured")
        layer_inventory = _layer_inventory(arena)
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        live_at_install = [
            int(g) for g, _n in arena.live_segs if g is not None]

        # THE QUESTION LENGTHS ARE CROSS-CHECKED against the registration
        # HERE, before any serve, through the arena's own encode + template.
        registered_q = registration[
            "question_lengths_REGISTERED_BEFORE_ANY_GATE"]
        for probe in plan:
            probe_id = str(probe["probe_id"])
            observed = _question_ntok(arena, str(probe["question"]))
            expected = registered_q.get(probe_id)
            entry = {
                "probe_id": probe_id,
                "observed_question_ntok": int(observed),
                "registered_question_ntok": (
                    None if expected is None
                    else int(expected["question_ntok"])),
            }
            entry["agree"] = bool(
                expected is not None
                and int(expected["question_ntok"]) == int(observed))
            question_ntok_checks.append(entry)
            if expected is not None and not entry["agree"]:
                raise RS4Error(
                    f"{probe_id}: the arena encodes its prompt to "
                    f"{observed} tokens but the registration says "
                    f"{expected['question_ntok']}; the row split would be "
                    "measuring a different question than the one registered")

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
                info["rs4_arm_note"] = (
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

                prelude = _prelude(repo, e2e, question, flags, arena,
                                   mount_ids)
                base = _snapshot_counterfactual(arena)
                native_base = _snapshot_native_ids(repo)
                missing = [v for v in mount_ids if int(v) >= len(arena.grafts)]
                if missing:
                    raise RS4Error(
                        f"{probe_id}/{arm}/{variant}: mount ids {missing} do "
                        f"not exist after the C0 prelude "
                        f"({len(arena.grafts)} grafts)")
                _clear_boat(arena)

                capture: dict[str, Any] | None = None
                fed_ntok = 0
                if arm == "C5":
                    feed_receipt = _feed_live(
                        repo, e2e, fixture,
                        [str(v) for v in item["live_node_ids"]])
                    fed_ntok = int(feed_receipt["live_tokens_after_feed"])
                    picks_requested = []
                elif arm in CAPTURE_ARMS:
                    capture = _recapture_under_pin(
                        repo, arena, read_rs3_registration(), probe_id, arm,
                        levers["capture_pin"], mount_ids)
                    picks_requested = [
                        int(v) for v in capture["fresh_graft_ids"]]
                else:
                    picks_requested = list(mount_ids)

                before_seat = getattr(arena, "_rs3_seat_explicit", None)
                try:
                    arena._rs3_seat_explicit = bool(levers["seat_near_live"])
                    answer, info, mounts, mass = _serve_direct(
                        repo, e2e, question, flags, picks_requested, fed_ntok)
                finally:
                    arena._rs3_seat_explicit = before_seat

                if capture is not None:
                    info["rs4_capture"] = capture
                info["rs4_c0_prelude"] = prelude
                info["rs4_levers"] = dict(levers)
                served.append(_row(
                    probe, probe_id, session_id, arm, variant, True, question,
                    answer, mounts, picks_requested, feed_receipt, mass, info,
                    int(info.get("rs4_cur_mount_n_at_serve", 0)), started,
                    levers))
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
        "phase": "RS4",
        "order": "orders/GRM_RS4_CEILING_REPARTITION.md",
        "arm": arm,
        "levers": levers,
        "session_id": session_id,
        "registered_probe_ids": sorted(targets),
        "why_a_rerun": (
            "RS3's receipts do not persist per-key-row attention: "
            "grm_demand._full_mass softmaxes and immediately sums into four "
            "scalars per layer, so the live band cannot be re-cut from disk."),
        "instrument": (
            "scripts/grm_rs4_ceiling_gpu.RowSplitMassObserver — a SUBCLASS of "
            "RS2's LayerTypeMassObserver that adds recording wrappers OUTSIDE "
            "the base class's own and modifies none of its arithmetic; the "
            "base partition is carried verbatim on every row."),
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
        "question_ntok_checks": question_ntok_checks,
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
        "rs3_registration": file_record(RS3_REGISTRATION_PATH),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_rs4_ceiling_gpu": file_record(Path(__file__).resolve()),
            "grm_rs4_row_split": file_record(
                ROOT / "scripts" / "grm_rs4_row_split.py"),
            "grm_rs4_registration": file_record(
                ROOT / "scripts" / "grm_rs4_registration.py"),
            "grm_rs3_capture_seat_gpu": file_record(
                ROOT / "scripts" / "grm_rs3_capture_seat_gpu.py"),
            "grm_rs2_mount_read_gpu": file_record(
                ROOT / "scripts" / "grm_rs2_mount_read_gpu.py"),
            "grm_rs1_read_strength_gpu": file_record(
                ROOT / "scripts" / "grm_rs1_read_strength_gpu.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
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
        raise RS4Error("--session-id is required for the arm command")
    if not args.arm:
        raise RS4Error("--arm is required for the arm command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_arm(str(args.session_id), str(args.arm))
    path = emit(payload, f"grm_rs4_{args.arm}_{args.session_id}")
    print(f"receipt={path}")
    for row in payload["probes"]:
        table = assemble_table([row])[0]
        print(json.dumps({
            "arm": table["arm"],
            "probe_id": table["probe_id"],
            "variant": table["variant"],
            "registered": row["registered_probe"],
            "correct": table["correct"],
            "mounted_full": table["mounted_mass_full_layers"],
            "live_full": table["live_mass_full_layers"],
            "prior_full": table["prior_live_mass_full_layers"],
            "fed_full": table["fed_text_mass_full_layers"],
            "question_full": table["question_mass_full_layers"],
            "answer_full": table["answer_mass_full_layers"],
            "subbands_ok": table["subbands_sum_to_live_full"],
            "partition_ok": table["partition_sums_to_one_full"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
