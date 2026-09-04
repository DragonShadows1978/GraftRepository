#!/usr/bin/env python3
"""GRM-RS2 — why is mounted K/V read ~4x weaker than the same text live?

ORDER: ``orders/GRM_RS2_MOUNT_READ_DEFICIT.md``.
REGISTRATION: ``artifacts/grm_rs2/registration.json`` (written FIRST, by
``scripts/grm_rs2_registration.py``; this module refuses to run without it).

THE QUESTION.  RS1 measured the gap and exonerated quantization: the exact node
text fed LIVE reads at live-band mass ~0.74 and answers; MOUNTED as a graft it
reads at mounted-band mass ~0.18 and refuses, with the identical mount id.  RS2
measures the candidate CAUSES of that gap.  MEASUREMENT ONLY: nothing under
``core/`` or ``config/`` is touched, no threshold is retuned, no production path
changes.

THE BUILD is RS1's, unchanged and IMPORTED, not re-implemented:
``grm_rs1_read_strength_gpu._load_lived_repo`` (which itself delegates to the
read-only ``grm_det1_2_gpu._load_model_repo``), then
``grm_det1_3_gpu._install_lived_nodes``.  The fixture's WHOLE probe plan is
replayed in the lived order for the same reason RS1 gave: serving a target probe
alone is a different computation and would not be a reproduction.

THE ARMS, and the exact seam each one needs.

  B0   the reproduction.  RS1's A0 path: the production probe ladder, unchanged.

  B1   H-CAPTURE, literal.  ``arena.deposit(capture_text)`` appends a FRESH
       graft; the probe is served over that id instead of the installed one.
       SEAM: none needed — ``deposit`` is production.
       STRUCTURAL FACT, registered before the run: under the SPEC frame the
       lived installer's ``arena.feed(text, deposit=True)`` takes
       ``ArenaCache.feed``'s ``if self.ephemeral:`` branch and calls
       ``self.deposit(text)``.  So the installed graft was ALREADY produced by
       this exact call.  The arm runs anyway and reports a PAYLOAD IDENTITY
       comparison (per-layer sha256 of the stored payload) — "the re-capture is
       bit-identical" is the finding, not a reason to skip the arm.

  B1b  H-CAPTURE, context varied.  The capture forward runs over
       ``encode(HARMONY_SINK) + encode(capture_text)`` and ONLY the
       capture_text rows are kept.
       SEAM: PRODUCTION LACKS THIS.  ``ArenaCache.deposit`` harvests the text
       alone; no production call harvests a text behind a prefix and stores
       only the text's rows.  ``_deposit_with_prefix`` below is that call,
       built out of the same ``arena._harvest`` production uses, so the only
       difference from ``deposit`` is the prefix and the row slice.

  B2   H-SCAFFOLD.  ``arena.deposit(scaffold_text)`` where scaffold_text is
       ``harmony_turn(user, None)`` — the UNANSWERED, question-shaped turn.
       The order's literal B2 ("wrapped exactly as A1 fed it live") is the
       installed capture text itself, so running it would be B1 again; the
       registration says so, and the arm varies the one scaffold axis that is
       actually free.

  B3a  H-BAND-POSITION, far end.   B3b  H-BAND-POSITION, near end.
       SEAM: PRODUCTION LACKS THIS.  At bootstrap ``ArenaCache._attempt``
       concatenates the sink payload and the mount payloads into ONE injection
       block, and ``GptOssAttentionTC.__call__`` RoPEs the whole block at
       ``cos.slice(0, 0, graft_seats)`` — positions 0.. — so a mount ALWAYS
       lands at positions [n_sink, n_sink + mount_ntok).  ``_positioned_injection``
       below pre-rotates the mount's keys by the delta, so after the layer
       applies its own RoPE the mount's net positions are the requested ones.
       The PHYSICAL cache rows are untouched (the mount stays a packed prefix);
       only the positional encoding moves.  B3b requests production's OWN
       offset through the same path, which is the control proving the path
       itself changes nothing.

  B4   H-SLIDING.  No serve of its own: the instrument records BOTH layer types
       on every arm and B4 is the read of the B0 and B5 rows split by type.
       SEAM: PRODUCTION LACKS THIS.  ``core.grm_demand.DemandObserver`` wraps
       ``sliding_sink_attention_tc`` solely to SUPPRESS it, and RS1's
       ``LayerMassObserver`` inherits that suppression.  ``LayerTypeMassObserver``
       below records the sliding layers instead, reproducing the sliding
       kernel's own [k0, k1) slice and causal+window mask for the last query
       row so the partition is the one the kernel actually computed.

  B5   the ceiling.  RS1's A1 seam, IMPORTED unchanged (``_feed_live``).

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

from core.grm_frame import ENV_NAME as PERSISTENT_BOAT_ENV  # noqa: E402
from core.grm_frame import env_persistent_boat  # noqa: E402
from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes)
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_rs1_read_strength_gpu import (  # noqa: E402
    MASS_NAMES, _feed_live, _load_lived_repo, _probe_plan, answer_verdict)
from scripts.grm_rs2_registration import RS2Error  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs2"
REGISTRATION = ARTIFACT_DIR / "registration.json"
SCHEMA_PREFIX = "grm.rs2"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 560
LOCK_WAIT_SECONDS = 7200

#: Every arm this module SERVES, in registration order.  B4 is a measurement
#: partition over B0's and B5's rows, not a serve, so it is not in this tuple.
ARMS = ("B0", "B1", "B1p", "B1b", "B2", "B3a", "B3b", "B5")

#: The arms whose mount set is a FRESH graft this module deposited.
CAPTURE_ARMS = ("B1", "B1p", "B1b", "B2")

#: The capture arms that harvest at the READ geometry — the query position a
#: served turn leaves (``n_sink + arena_width``) rather than the position the
#: lived installer captured at (0).  See ``_capture_and_mount``'s docstring:
#: ``ArenaCache.deposit`` does not pin this, so RS2 makes it an arm variable.
READ_GEOMETRY_ARMS = ("B1p",)

#: The arms that vary the mount's RoPE positions inside the arena band.
POSITION_ARMS = ("B3a", "B3b")

#: The two layer types the extended instrument partitions by.  Read off the
#: model's own config through ``self_attn.layer_type``; never a layer index.
FULL = "full_attention"
SLIDING = "sliding_attention"
LAYER_TYPES = (FULL, SLIDING)


# ======================================================================
# The extended instrument (B4's seam)
# ======================================================================
class LayerTypeMassObserver:
    """RS1's mass instrument, extended to record the SLIDING layers too.

    RS1 wrapped ``core.grm_demand.DemandObserver`` and kept the per-layer rows
    it already computed.  Those rows are FULL-ATTENTION ONLY: the production
    observer's ``sliding_wrapper`` exists purely to set ``_suppress_sink`` so
    the sliding layers' inner ``sink_attention_tc`` calls are not counted.
    That suppression is inherited from the frozen DET1 race observer and it is
    correct FOR D-NGH, whose registered score is defined on the full band.  It
    is exactly what B4 needs removed.

    THE FULL-LAYER ARITHMETIC IS UNCHANGED.  The production observer still
    computes the full-attention rows, still through its own ``_full_mass``,
    still checked by its own partition law — so this arm's full-layer numbers
    are directly comparable to RS1's and to a production demand receipt.  The
    sliding rows are computed ALONGSIDE, by this class, from the sliding
    kernel's own arguments.

    THE SLIDING PARTITION, as registered.  ``sliding_sink_attention_tc`` does
    not score all S key rows.  For the last query row (``q_abs = S - 1``) it
    slices keys to ``[k0, k1)`` with ``k0 = max(0, q_abs - window + 1)`` and
    ``k1 = min(S, q_abs + 1)``, then masks ``k_abs <= q_abs and k_abs >
    q_abs - window`` inside that slice.  This class reproduces that slice and
    that mask for the last query row only, softmaxes ``[scores | sink logit]``
    exactly as the kernel does, and sums into the same four bands by ABSOLUTE
    cache-row index.  Rows outside ``[k0, k1)`` contribute exactly zero,
    because the kernel never scores them — a structural zero, recorded as such
    next to ``window_reaches_mount`` and the intersection width.
    """

    def __init__(self, arena: Any, ngen: int, threshold: float) -> None:
        from core import grm_demand

        self.arena = arena
        self._inner = grm_demand.DemandObserver(
            arena, int(ngen), float(threshold), early_abort=False)
        self.full_rows: list[list[dict[str, float]]] = []
        self.sliding_rows: list[list[dict[str, Any]]] = []
        self._pending_sliding: list[dict[str, Any]] = []
        self._gpt = None
        self._original_sliding = None
        self._original_forward = None
        self.expected_sliding_layers = 0
        self.geometry: dict[str, Any] = {}

        original_summarize = self._inner._summarize_mass

        def summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
            # BEFORE the delegation, because the base class's partition check
            # RAISES and a raise must not cost us the rows that show why.
            self.full_rows.append([
                {str(k): float(v) for k, v in dict(row).items()}
                for row in rows
            ])
            return original_summarize(rows)

        self._inner._summarize_mass = summarize  # type: ignore[method-assign]

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "LayerTypeMassObserver":
        from core import gpt_oss20b_tc as gpt

        self._inner.__enter__()
        # The production observer has now installed ITS sliding wrapper (which
        # suppresses) and ITS forward wrapper.  Layer OURS on top of both: the
        # sliding one records and then delegates to the suppressing one, so the
        # inner sink calls stay suppressed and the full-layer count the base
        # class enforces is unchanged.
        self._gpt = gpt
        self._original_sliding = gpt.sliding_sink_attention_tc
        self._original_forward = self.arena._forward
        observer = self

        layers = list(
            getattr(getattr(self.arena, "m", None), "layers", ()) or ())
        self.expected_sliding_layers = sum(
            1 for layer in layers
            if str(getattr(layer.self_attn, "layer_type", "")) == SLIDING)
        if not self.expected_sliding_layers:
            raise RS2Error(
                "no sliding-attention layers found on this model; the RS2 "
                "instrument is defined on the GPT-OSS two-layer-type path")
        self.geometry = {
            "n_sink": int(self.arena.n_sink),
            "arena_width": int(self.arena.width),
            "live_shift": int(self.arena.live_shift),
            "layer_counts": {
                FULL: sum(
                    1 for layer in layers
                    if str(getattr(layer.self_attn, "layer_type", "")) == FULL),
                SLIDING: int(self.expected_sliding_layers),
            },
            "sliding_window": int(next(
                int(layer.self_attn.sliding_window) for layer in layers
                if str(getattr(layer.self_attn, "layer_type", "")) == SLIDING)),
        }

        def sliding_wrapper(query, key, value, sinks, *, scale,
                            sliding_window, num_heads_per_kv=1, attn_block=128,
                            allowed_mask=None):
            result = observer._original_sliding(
                query, key, value, sinks,
                scale=scale,
                sliding_window=sliding_window,
                num_heads_per_kv=num_heads_per_kv,
                attn_block=attn_block,
                allowed_mask=allowed_mask,
            )
            if observer._inner._inside_forward:
                observer._pending_sliding.append(observer._sliding_mass(
                    query, key, sinks,
                    scale=float(scale),
                    sliding_window=int(sliding_window),
                    num_heads_per_kv=int(num_heads_per_kv),
                    allowed_mask=allowed_mask,
                ))
            return result

        def forward_wrapper(ids, last_only=True):
            observer._pending_sliding = []
            try:
                return observer._original_forward(ids, last_only=last_only)
            finally:
                # Closed on the way out whether or not the inner forward raised,
                # so the sliding rows stay aligned with the full ones the
                # production observer appended inside the same call.
                observer.sliding_rows.append(list(observer._pending_sliding))
                observer._pending_sliding = []

        gpt.sliding_sink_attention_tc = sliding_wrapper
        self.arena._forward = forward_wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_forward is not None:
            self.arena._forward = self._original_forward
        if self._gpt is not None and self._original_sliding is not None:
            self._gpt.sliding_sink_attention_tc = self._original_sliding
        return self._inner.__exit__(exc_type, exc, tb)

    # -- the sliding score ----------------------------------------------
    def _sliding_mass(self, query, key, sinks, *, scale: float,
                      sliding_window: int, num_heads_per_kv: int,
                      allowed_mask) -> dict[str, Any]:
        """The registered four-band partition on ONE sliding layer's last row.

        Deliberately built out of the SAME operations
        ``DemandObserver._full_mass`` uses (``gpt._repeat_kv``, ``tc.matmul``
        with ``trans_b``, ``tc.cat`` of the sink logit, ``softmax(-1)``), so
        the only intended difference from the full-layer computation is the key
        SLICE and the window MASK the sliding kernel itself applies.
        """
        gpt = self._gpt
        L = int(query.shape[2])
        S = int(key.shape[2])
        window = max(1, int(sliding_window))
        q_abs = S - 1                       # the LAST query row, absolute
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
                raise RS2Error(
                    "sliding allowed-mask shape mismatch: "
                    f"expected={(1, 1, L, S)} observed={full_allowed.shape}")
            allowed = full_allowed[0, 0, L - 1, k0:k1].astype(bool, copy=False)
        mask = np.where(allowed, 0.0, -1.0e4).astype(np.float32)
        mask_t = gpt.tc.tensor(
            mask.reshape(1, 1, 1, k1 - k0), dtype="float32").astype(query.dtype)
        scores = scores + mask_t

        batch, heads, _one, _source = (int(scores.shape[i]) for i in range(4))
        sink_logits = sinks.reshape([1, heads, 1, 1]).expand([batch, heads, 1, 1])
        probs = gpt.tc.cat([scores, sink_logits], dim=-1).softmax(-1)
        values = np.asarray(probs.float().numpy(), dtype=np.float32)[0, :, 0, :]

        # The four bands, by ABSOLUTE cache row, intersected with [k0, k1).
        # A band the window cannot reach contributes exactly zero, because the
        # kernel never scored those rows — a STRUCTURAL zero, flagged as one.
        sink_end = int(self.arena.n_sink)
        mount_end = min(S, sink_end + int(self.arena.cur_mount_n))
        scored = k1 - k0

        def band(lo: int, hi: int) -> tuple[float, int]:
            lo_c, hi_c = max(lo, k0), min(hi, k1)
            if hi_c <= lo_c:
                return 0.0, 0
            return (float(np.mean(
                values[:, lo_c - k0:hi_c - k0].sum(axis=1))), hi_c - lo_c)

        sink_mass, sink_rows = band(0, sink_end)
        mounted_mass, mount_rows = band(sink_end, mount_end)
        live_mass, live_rows = band(mount_end, S)
        learned = float(np.mean(values[:, scored]))

        return {
            "physical_sink_mass": sink_mass,
            "mounted_mass": mounted_mass,
            "live_mass": live_mass,
            "learned_sink_mass": learned,
            "S": int(S),
            "q_abs": int(q_abs),
            "k0": int(k0),
            "k1": int(k1),
            "sliding_window": int(window),
            "sink_band": [0, int(sink_end)],
            "mount_band": [int(sink_end), int(mount_end)],
            "live_band": [int(mount_end), int(S)],
            "rows_scored": int(scored),
            "sink_rows_in_window": int(sink_rows),
            "mount_rows_in_window": int(mount_rows),
            "live_rows_in_window": int(live_rows),
            "window_reaches_mount": bool(mount_rows > 0),
            "window_reaches_sink": bool(sink_rows > 0),
        }

    # -- the tables -----------------------------------------------------
    def rows(self) -> list[dict[str, Any]]:
        """The answer-position rows, under the race's flush-drop convention."""
        return self._inner.finish()

    def partition(self) -> dict[str, Any]:
        """The RS2 table for ONE serve: both layer types, side by side."""
        kept = self.rows()
        n = len(kept)
        full_steps = self.full_rows[:n]
        sliding_steps = self.sliding_rows[:n]

        by_type = {
            name: _summarize_layer_type(name, steps)
            for name, steps in ((FULL, full_steps), (SLIDING, sliding_steps))
        }

        # The FULL-layer means are ALSO taken straight off the production
        # observer's own kept rows, so the number this receipt reports and the
        # number a production demand receipt would report are one number, not
        # two computations that happen to agree.
        production_mean = {
            key: (float(np.mean([float(row[key]) for row in kept]))
                  if kept else None)
            for key in MASS_NAMES
        }
        recomputed = by_type[FULL]["mean_over_answer_positions"]
        agreement = {
            key: (
                None if production_mean[key] is None
                or recomputed.get(key) is None
                else abs(float(production_mean[key]) - float(recomputed[key])))
            for key in MASS_NAMES
        }
        return {
            "answer_positions": n,
            "geometry": dict(self.geometry),
            "by_layer_type": by_type,
            "production_full_layer_mean": production_mean,
            "production_vs_recomputed_abs_delta": agreement,
            "per_token_mounted_mass_full": [
                float(row["mounted_mass"]) for row in kept],
            "per_token_mounted_mass_sliding": [
                (float(np.mean([float(r["mounted_mass"]) for r in step]))
                 if step else None)
                for step in sliding_steps
            ],
            "sliding_window_reach": _window_reach(sliding_steps),
        }


def _summarize_layer_type(
    layer_type: str, steps: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Mean over answer positions of the per-layer mean, for ONE layer type.

    Pure over already-captured rows so the CPU tests can pin the arithmetic —
    including the partition law — with no GPU.
    """
    steps = [list(step) for step in steps if step]
    if not steps:
        return {
            "layer_type": layer_type,
            "layers_observed": 0,
            "answer_positions": 0,
            "mean_over_answer_positions": {name: None for name in MASS_NAMES},
            "per_layer_mean": [],
            "top_contributing_layer": None,
            "partition_sum": None,
            "partition_sums_to_one": None,
        }
    width = len(steps[0])
    for step in steps:
        if len(step) != width:
            raise RS2Error(
                f"{layer_type}: layer capture count drifted between answer "
                f"positions ({width} then {len(step)})")
    mean = {
        name: float(np.mean([
            float(np.mean([float(row[name]) for row in step]))
            for step in steps]))
        for name in MASS_NAMES
    }
    per_layer: list[dict[str, Any]] = []
    for ordinal in range(width):
        row: dict[str, Any] = {
            name: float(np.mean([float(step[ordinal][name]) for step in steps]))
            for name in MASS_NAMES
        }
        row["layer_ordinal"] = int(ordinal)
        row["layer_type"] = layer_type
        per_layer.append(row)
    top = max(per_layer, key=lambda r: float(r["mounted_mass"]))
    return {
        "layer_type": layer_type,
        "layers_observed": int(width),
        "answer_positions": len(steps),
        "mean_over_answer_positions": mean,
        "per_layer_mean": per_layer,
        "top_contributing_layer": {
            "layer_ordinal": int(top["layer_ordinal"]),
            "mounted_mass": float(top["mounted_mass"]),
        },
        "partition_sum": float(sum(float(v) for v in mean.values())),
        "partition_sums_to_one": bool(partition_sum_ok(mean)),
    }


def partition_sum_ok(mean: Mapping[str, Any]) -> bool:
    """The partition arithmetic check, exposed for the CPU tests.

    Four masses summing to one within the tolerance
    ``grm_demand._summarize_mass`` enforces.  A ``None`` means "no answer
    positions were captured", which is not a partition at all.
    """
    values = [mean.get(name) for name in MASS_NAMES]
    if any(value is None for value in values):
        return False
    return 0.98 <= float(sum(float(v) for v in values)) <= 1.02


def _window_reach(
    steps: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Whether, and by how much, the sliding window reached the mount band.

    The registered definition says a structural zero must be distinguishable
    from a measured near-zero, so the reach is REPORTED rather than inferred
    from the mass being small.  Pure over captured rows (CPU tested).
    """
    flat = [dict(row) for step in steps for row in step]
    if not flat:
        return {
            "observed_rows": 0,
            "always_reaches_mount": None,
            "never_reaches_mount": None,
        }
    reaches = [bool(row["window_reaches_mount"]) for row in flat]
    mount_rows = [int(row["mount_rows_in_window"]) for row in flat]
    widths = [
        int(row["mount_band"][1]) - int(row["mount_band"][0]) for row in flat]
    visible = [(m / w) if w else 0.0 for m, w in zip(mount_rows, widths)]
    return {
        "observed_rows": len(flat),
        "always_reaches_mount": bool(all(reaches)),
        "never_reaches_mount": bool(not any(reaches)),
        "reaching_rows": int(sum(1 for value in reaches if value)),
        "mount_rows_in_window_min": int(min(mount_rows)),
        "mount_rows_in_window_max": int(max(mount_rows)),
        "mount_band_width_min": int(min(widths)),
        "mount_band_width_max": int(max(widths)),
        "fraction_of_mount_band_visible_min": float(min(visible)),
        "fraction_of_mount_band_visible_max": float(max(visible)),
        "S_min": int(min(int(row["S"]) for row in flat)),
        "S_max": int(max(int(row["S"]) for row in flat)),
        "k0_min": int(min(int(row["k0"]) for row in flat)),
        "k0_max": int(max(int(row["k0"]) for row in flat)),
        "sliding_window": int(flat[0]["sliding_window"]),
    }


# ======================================================================
# Registration reading + the arm plan (pure — CPU tested)
# ======================================================================
def read_registration(path: Path = REGISTRATION) -> dict[str, Any]:
    if not Path(path).exists():
        raise RS2Error(
            f"registration missing at {path}. It is written BEFORE any gate "
            "run by scripts/grm_rs2_registration.py; this harness refuses to "
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


def solace_variant_applies(registration: Mapping[str, Any],
                           probe_id: str) -> bool:
    """Does the order's extra solace-fact run apply to this probe?"""
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    return str(variant["probe_id"]) == str(probe_id)


def arm_plan(registration: Mapping[str, Any],
             session_id: str) -> list[dict[str, Any]]:
    """The (arm, probe, variant) work items for one fixture.

    Pure: no GPU, no model, no filesystem beyond the registration already read.
    Every arm runs on every registered probe of the fixture; the solace probe
    additionally gets a ``solace_fact`` variant of every arm, which is what the
    order's "ALSO run every arm with the solace FACT node mounted" asks for.
    """
    probes = registered_probes(registration)
    ids = probes_for_session(registration, session_id)
    if not ids:
        raise RS2Error(f"no registered probe belongs to fixture {session_id}")
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    plan: list[dict[str, Any]] = []
    for arm in ARMS:
        for probe_id in ids:
            row = probes[probe_id]
            plan.append({
                "arm": arm,
                "probe_id": probe_id,
                "session_id": session_id,
                "variant": "registered",
                "role": str(row["role"]),
                "mount_ids": [int(v) for v in row["a0_mounted_ids"]],
                "live_ids": [int(v) for v in row["live_ids"]],
                "live_node_ids": [str(v) for v in row["live_node_ids"]],
            })
            if solace_variant_applies(registration, probe_id):
                plan.append({
                    "arm": arm,
                    "probe_id": probe_id,
                    "session_id": session_id,
                    "variant": "solace_fact",
                    "role": str(row["role"]),
                    "mount_ids": [int(variant["graft_id"])],
                    "live_ids": [int(v) for v in row["live_ids"]],
                    "live_node_ids": [str(v) for v in row["live_node_ids"]],
                })
    return plan


def capture_text_for(registration: Mapping[str, Any], probe_id: str,
                     graft_id: int, arm: str) -> dict[str, Any] | None:
    """The registered capture text this arm re-captures, or ``None``.

    Only B2 varies the TEXT (the unanswered question-shaped scaffold); B1, B1p
    and B1b all re-capture the installed capture text and vary the capture
    CONTEXT or the capture POSITION instead, so they share one text.

    Returns ``None`` when the mounted id is NOT a fixture node (a fit-time
    split child): the registration records that such an id has no fixture
    capture text and the arm falls back to the graft's OWN stored text, which
    only the GPU run can read.  Pure otherwise, so the CPU tests can pin the
    selection rule without a repository.
    """
    row = registered_probes(registration)[probe_id]
    variant = registration["solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]
    if (str(variant["probe_id"]) == str(probe_id)
            and int(variant["graft_id"]) == int(graft_id)):
        text = (str(variant["scaffold_text"]) if arm == "B2"
                else str(variant["capture_text"]))
        return {
            "source": "solace_fact_variant",
            "node_id": str(variant["node_id"]),
            "text": text,
        }
    for node in row["capture_nodes"]:
        if int(node["graft_id"]) != int(graft_id):
            continue
        if not node.get("is_fixture_node"):
            return None
        text = (str(node["capture_text_B2_scaffold"]) if arm == "B2"
                else str(node["capture_text_B0_B1"]))
        return {
            "source": "registered_capture_node",
            "node_id": str(node["node_id"]),
            "text": text,
        }
    return None


# ======================================================================
# Payload identity (B1's finding)
# ======================================================================
def payload_digest(graft: Mapping[str, Any]) -> dict[str, Any]:
    """A per-layer sha256 of a graft's stored payload.

    B1 claims to be a re-capture of a graft that was itself captured by the
    same call.  "Bit-identical" is a claim about bytes, so it is checked
    against bytes: every layer's payload tensors are pulled to host and hashed,
    and the digest of the concatenation is what the receipt carries.
    """
    payload = graft.get("h")
    if payload is None:
        return {"available": False, "reason": "graft payload is not resident"}
    digest = hashlib.sha256()
    per_layer: list[str] = []
    for layer in payload:
        layer_digest = hashlib.sha256()
        for key in sorted(layer):
            array = layer[key]
            array = array if isinstance(array, np.ndarray) else array.numpy()
            layer_digest.update(str(key).encode("utf-8"))
            layer_digest.update(np.ascontiguousarray(array).tobytes())
        value = layer_digest.hexdigest()
        per_layer.append(value)
        digest.update(value.encode("utf-8"))
    return {
        "available": True,
        "layers": len(per_layer),
        "sha256": digest.hexdigest(),
        "per_layer_sha256": per_layer,
        "ntok": int(graft.get("ntok", 0)),
    }


# ======================================================================
# B1b's seam — a prefix-context capture production does not have
# ======================================================================
def _deposit_with_prefix(arena, prefix_text: str, text: str) -> dict[str, Any]:
    """Harvest ``text`` behind ``prefix_text`` and deposit ONLY the text rows.

    PRODUCTION LACKS THIS.  ``ArenaCache.deposit`` runs ``self._harvest(
    self.encode(text))`` — the text alone.  H-CAPTURE asks what happens when
    the capture-time CONTEXT differs, so this runs the SAME harvest over
    ``encode(prefix) + encode(text)`` and keeps the last ``len(encode(text))``
    rows of every layer.  Everything else about the deposited graft — the
    routing key, the ntok, the device residency, the epoch bump — comes from
    the same expressions ``deposit`` uses, so the only difference between this
    graft and a ``deposit`` graft is the context the keys were computed in.

    The graft's ROUTING KEY is deliberately taken from the arena's own
    ``_node_key(text)`` on the BARE text: routing is not the axis under test,
    and a context-polluted centroid would move the mount set as well as the
    read strength, which would confound the arm.  ``deposit_from_cache``'s own
    docstring records the same reasoning for the same reason.
    """
    from core.mistral7b_tc import tc

    prefix_ids = [int(v) for v in arena.encode(prefix_text)]
    text_ids = [int(v) for v in arena.encode(text)]
    if not text_ids:
        raise RS2Error("B1b capture text encodes to zero tokens")
    combined = prefix_ids + text_ids
    harvest = arena._harvest(combined)
    n_text = len(text_ids)
    device: list[dict[str, Any]] = []
    for layer in harvest:
        row: dict[str, Any] = {}
        for key, dim in arena.PAYLOAD:
            array = layer[key]
            array = array if isinstance(array, np.ndarray) else array.numpy()
            total = int(array.shape[dim])
            if total != len(combined):
                raise RS2Error(
                    f"B1b harvest row count {total} != prefix+text "
                    f"{len(combined)} on payload key {key!r}")
            index: list[Any] = [slice(None)] * array.ndim
            index[dim] = slice(total - n_text, total)
            row[key] = tc.tensor(
                np.ascontiguousarray(array[tuple(index)])).astype(arena.dt)
        device.append(row)
    arena.grafts.append({
        "h": device,
        "cent": arena._node_key(text),
        "ntok": n_text,
        "text": text,
    })
    arena._bump_cuda_gqa_epoch()
    index_new = len(arena.grafts) - 1
    return {
        "graft_id": int(index_new),
        "call": "scripts/grm_rs2_mount_read_gpu._deposit_with_prefix",
        "prefix_text": str(prefix_text),
        "prefix_ntok": len(prefix_ids),
        "text_ntok": n_text,
        "harvest_ntok": len(combined),
        "rows_kept": f"last {n_text} of {len(combined)}",
        "routing_key_from": "arena._node_key(text) on the BARE text",
        "seam": (
            "PRODUCTION LACKS THIS: ArenaCache.deposit harvests the text "
            "alone; no production call harvests behind a prefix and stores "
            "only the text's rows"),
    }


# ======================================================================
# B3's seam — seating the mount at a chosen band offset
# ======================================================================
def _positioned_injection(arena, picks: Sequence[int],
                          mount_pos0: int) -> dict[str, Any]:
    """Install a bootstrap injection whose MOUNT lands at ``mount_pos0``.

    PRODUCTION LACKS THIS.  ``ArenaCache._attempt``'s bootstrap branch
    concatenates the sink payload and the mount payloads into ONE block, and
    ``GptOssAttentionTC.__call__`` RoPEs the whole block at
    ``cos.slice(0, 0, graft_seats)``.  The sink therefore lands at positions
    [0, n_sink) and a mount ALWAYS lands at [n_sink, n_sink + mount_ntok).
    Nothing under ``core/`` moves it.

    THE MECHANISM, and why it is a position change and nothing else.  The layer
    will apply RoPE at absolute positions ``[0, graft_seats)`` no matter what.
    The mount occupies block rows ``[n_sink, n_sink + n)``, so the layer will
    rotate those rows by ``n_sink + j`` for j in [0, n).  To land them at
    ``mount_pos0 + j`` the payload is pre-rotated here by
    ``delta = mount_pos0 - n_sink`` — one extra rotation composed with the
    layer's, which is exactly the composition ``ArenaCache._rope_block_at``
    already relies on for re-seating.  Only the KEY carries position; the value
    payload is untouched, and the PHYSICAL cache rows are unchanged (the mount
    stays a packed prefix immediately after the sink).  So the arm varies the
    mount's positional encoding and nothing else.

    The sink is left where production puts it: moving it would change a second
    variable.
    """
    from core import kv_graft

    picks = [int(v) for v in picks]
    n_sink = int(arena.n_sink)
    mount_ntok = sum(int(arena.grafts[i]["ntok"]) for i in picks)
    delta = int(mount_pos0) - n_sink
    rope_key = arena.PAYLOAD[0][0]

    mounts = [{"h": arena.sink_h}] + [arena.grafts[i] for i in picks]
    injection: list[dict[str, Any]] = []
    for layer_index in range(len(arena.m.layers)):
        block: dict[str, Any] = {}
        for key, dim in arena.PAYLOAD:
            parts = []
            for position, graft in enumerate(mounts):
                array = graft["h"][layer_index][key]
                array = (array if isinstance(array, np.ndarray)
                         else array.numpy())
                if position and delta and key == rope_key:
                    # The MOUNT rows only, and only the rope-carrying payload.
                    array = _rotate_host(arena, array, dim, delta)
                parts.append(array)
            block[key] = np.concatenate(parts, axis=dim)
        injection.append(block)
    kv_graft.set_injection(arena.m, injection)
    arena.cur_mounts = list(picks)
    arena.cur_mount_n = int(mount_ntok)
    return {
        "requested_mount_pos0": int(mount_pos0),
        "production_mount_pos0": n_sink,
        "delta_positions": int(delta),
        "mount_ntok": int(mount_ntok),
        "n_sink": n_sink,
        "live_shift": int(arena.live_shift),
        "mount_positions": [int(mount_pos0), int(mount_pos0) + int(mount_ntok)],
        "sink_positions": [0, n_sink],
        "physical_cache_rows_unchanged": True,
        "rope_payload_key": str(rope_key),
        "seam": (
            "PRODUCTION LACKS THIS: the bootstrap injection block is RoPE'd at "
            "cos.slice(0, 0, graft_seats), so a mount always lands at "
            "[n_sink, n_sink + mount_ntok)"),
    }


def _rotate_host(arena, array: np.ndarray, dim: int, delta: int) -> np.ndarray:
    """Pre-rotate a host payload by ``delta`` positions, on the seq axis.

    Uses the arena's OWN ``_rope_tensor`` (which is ``tc.rope_apply`` when the
    engine exposes it, else the ``F.apply_rotary`` fallback), so the rotation is
    the operation production composes with rather than a re-derivation of RoPE.
    A negative delta is applied as the inverse rotation of ``|delta|``, which is
    what ``_rope_block_at(..., inverse=True)`` does for the same reason.
    """
    from core.mistral7b_tc import tc

    if not delta:
        return array
    seq_axis = array.ndim - 2
    moved = array if dim == seq_axis else np.moveaxis(array, dim, seq_axis)
    tensor = tc.tensor(np.ascontiguousarray(moved)).astype(arena.dt)
    rotated = arena._rope_tensor(
        tensor, abs(int(delta)), inverse=bool(delta < 0),
        pair_swap=arena.ROPE_PAIR_SWAP)
    out = np.asarray(rotated.float().numpy()).astype(array.dtype)
    if dim != seq_axis:
        out = np.moveaxis(out, seq_axis, dim)
    return np.ascontiguousarray(out)


# ======================================================================
# The serves
# ======================================================================
def _serve_ladder(repo, e2e, question: str, flags: Mapping[str, Any],
                  ) -> tuple[str, dict[str, Any], list[int], dict[str, Any]]:
    """B0: the PRODUCTION probe path, observed with the extended instrument."""
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
    """B1/B1b/B2/B5: ONE generation attempt over an EXPLICIT mount set.

    RS1's ``_serve_direct``, with the extended instrument in place of the
    full-layers-only one and nothing else changed: the same
    ``e2e._budget_fit_mounts`` runs first (``_attempt``'s bootstrap branch has
    no width check because production never reaches it unfitted), the same
    ``arena._attempt`` call serves, and what the fit dropped is REPORTED rather
    than silently downgraded.
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
    out["rs2_grounded"] = bool(grounded)
    out["rs2_serving_path"] = "ArenaCache._attempt (routing disabled)"
    out["rs2_fit_requested"] = requested
    out["rs2_fit_seated"] = fitted
    out["rs2_fit_dropped"] = dropped
    out["rs2_arena_width"] = int(arena.width)
    out["rs2_cur_mount_n_at_serve"] = int(arena.cur_mount_n)
    out["rs2_live_segs_at_serve"] = [
        {"graft_id": (None if g is None else int(g)), "ntok": int(n)}
        for g, n in arena.live_segs]
    return str(answer), out, mounts, observer.partition()


def _serve_positioned(repo, e2e, question: str, flags: Mapping[str, Any],
                      picks: Sequence[int], mount_pos0: int,
                      ) -> tuple[str, dict[str, Any], list[int],
                                 dict[str, Any]]:
    """B3a/B3b: serve over a mount seated at a CHOSEN band offset.

    ``_attempt`` cannot be used here: its bootstrap branch would install
    PRODUCTION's injection and overwrite the arm's positioned one, and its
    non-bootstrap branch would ``swap`` the mount back to production's seats.
    So the arm drives the same three steps ``_attempt`` drives — install the
    injection, run the prompt forward (which consumes it and builds the cache),
    clear the injection, greedy-decode with the same early-stop rule and the
    same final commit forward — with the positioned injection in place of
    production's.  Everything else (the prompt template, the stop sequences,
    the greedy argmax) is production's own.
    """
    from core import grm_demand
    from core import kv_graft

    arena = repo.arena
    requested = [int(v) for v in picks]
    fitted = [int(v) for v in e2e._budget_fit_mounts(arena, requested)]
    dropped = [v for v in requested if v not in set(fitted)]
    if not fitted:
        raise RS2Error(
            f"the positioned arm has nothing to seat: requested={requested}")

    arena._ensure_h(fitted)
    seating = _positioned_injection(arena, fitted, int(mount_pos0))
    stops = arena.stop_sequences or ()

    with LayerTypeMassObserver(
        arena, int(flags["ngen"]), grm_demand.registered_threshold(),
    ) as observer:
        prompt_ids = arena.encode(arena._format_step_prompt(question))
        row = arena._forward(prompt_ids)
        kv_graft.clear_injection(arena.m)   # the injection fires once
        answer_ids = [int(row.argmax())]
        stopped = False
        for _ in range(int(flags["ngen"]) - 1):
            if any(stop in arena.decode(answer_ids) for stop in stops):
                stopped = True
                break
            row = arena._forward([answer_ids[-1]])
            answer_ids.append(int(row.argmax()))
        if not stopped and not any(
                stop in arena.decode(answer_ids) for stop in stops):
            # Commit the last predicted token, exactly as _attempt does.
            arena._forward([answer_ids[-1]])
    answer = arena.decode(answer_ids)
    for stop in stops:
        if stop in answer:
            answer = answer.split(stop, 1)[0]
    mounts = [int(v) for v in arena.cur_mounts]
    grounded, _c = arena._grounding_attribution(str(answer), mounts, question)
    info = {
        "rs2_grounded": bool(grounded),
        "rs2_serving_path": (
            "positioned bootstrap injection + arena._forward decode "
            "(routing disabled)"),
        "rs2_fit_requested": requested,
        "rs2_fit_seated": fitted,
        "rs2_fit_dropped": dropped,
        "rs2_arena_width": int(arena.width),
        "rs2_cur_mount_n_at_serve": int(arena.cur_mount_n),
        "rs2_seating": seating,
        "rs2_prompt_ntok": len(prompt_ids),
        "rs2_answer_ntok": len(answer_ids),
        "rs2_decode_note": (
            "ArenaCache._attempt's own decode loop, inlined because _attempt "
            "would re-install production's bootstrap injection and overwrite "
            "the arm's positioned one. Greedy argmax, the same early-stop "
            "rule, the same final commit forward."),
    }
    return str(answer), info, mounts, observer.partition()


# ======================================================================
# The arm runner
# ======================================================================
def _layer_inventory(arena) -> dict[str, Any]:
    """Which layers are which type, read off the model, never typed."""
    layers = list(arena.m.layers)
    by_type: dict[str, list[int]] = {FULL: [], SLIDING: []}
    for index, layer in enumerate(layers):
        name = str(getattr(layer.self_attn, "layer_type", ""))
        by_type.setdefault(name, []).append(int(index))
    windows = sorted({
        int(layer.self_attn.sliding_window) for layer in layers
        if getattr(layer.self_attn, "sliding_window", None) is not None})
    return {
        "layers": len(layers),
        "indices_by_type": by_type,
        "counts_by_type": {k: len(v) for k, v in by_type.items()},
        "sliding_windows_observed": windows,
        "n_sink": int(arena.n_sink),
        "arena_width": int(arena.width),
        "live_shift": int(arena.live_shift),
    }


def _clear_boat(arena) -> None:
    """The empty boat the spec frame's own turn-open leaves."""
    arena.caches, arena.pos, arena.live_segs = None, 0, []
    arena.cur_mounts, arena.cur_mount_n = [], 0
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift


def _snapshot_native_ids(repo) -> dict[int, int] | None:
    """The repository's graft-index -> native-node-id map, copied.

    ``grm_det1_e2e._restore_counterfactual`` truncates ``arena.grafts`` back to
    the snapshot length but does NOT touch ``GraftRepository._native_node_ids``.
    That is correct for RS1, whose arms never appended a graft.  RS2's capture
    arms DO append one per probe, so without this the map keeps an entry for an
    index that no longer exists and the NEXT probe's fresh graft — which reuses
    that index — is seen by ``_native_sync_node`` as already synced.  It then
    returns the stale id without setting ``graft['native_node_id']``, and the
    serve dies in ``_commit_native_mount`` -> ``_native_mount_ids`` with
    "graft N has no native_node_id".  MEASURED: that is exactly how the first
    B1 run on ``fresh_fact_controls`` failed, on graft 5, after the harness had
    already appended and rolled back one fresh graft on the previous probe.
    """
    mapping = getattr(repo, "_native_node_ids", None)
    if mapping is None:
        return None
    return {int(k): int(v) for k, v in mapping.items()}


def _restore_native_ids(repo, snapshot: dict[int, int] | None) -> None:
    """Put the native-id map back exactly as the arm found it."""
    if snapshot is None:
        return
    mapping = getattr(repo, "_native_node_ids", None)
    if mapping is None:
        return
    mapping.clear()
    mapping.update({int(k): int(v) for k, v in snapshot.items()})


def _prelude(repo, e2e, question: str, flags: Mapping[str, Any], arena,
             mount_ids: Sequence[int]) -> dict[str, Any]:
    """B0's own ladder, run first so the repository is the one B0 had here.

    RS1 measured why this is not optional: two of the registered mount ids are
    FIT-TIME SPLIT CHILDREN that do not exist until a fit has run, and the
    split is PERSISTED, so requesting them on a freshly installed repository
    raises out of ``_budget_fit_mounts``.
    """
    answer, info, mounts, _mass = _serve_ladder(repo, e2e, question, flags)
    return {
        "why": (
            "B0's own ladder was run first so this arm's repository is the one "
            "B0 had at this probe — including any PERSISTED fit-time split "
            "whose child the registered mount ids name"),
        "served_answer": str(answer),
        "mounted_ids": [int(v) for v in mounts],
        "fit_split_parent": info.get("fit_split_parent"),
        "fit_split_children": info.get("fit_split_children"),
        "fit_split_ephemeral": info.get("fit_split_ephemeral"),
        "grafts_after_prelude": len(arena.grafts),
        "arm_mount_ids": [int(v) for v in mount_ids],
    }


def _capture_and_mount(repo, arena, registration: Mapping[str, Any],
                       probe_id: str, arm: str,
                       mount_ids: Sequence[int]) -> dict[str, Any]:
    """Re-capture the arm's node(s) and return the fresh graft ids.

    The capture TEXT comes from the registration when the mounted id is a
    fixture node.  When it is a FIT-TIME SPLIT CHILD the registration says so,
    and records that the arm falls back to the graft's own stored text — the
    only text that child has, and the one a re-capture of that child must use.

    THE NATIVE SYNC IS NOT OPTIONAL, and it is production's own call.  MEASURED
    on the first B1 run, with a log: a graft appended straight onto
    ``arena.grafts`` (which is what ``ArenaCache.deposit`` does) carries no
    ``native_node_id``, so the first ``_attempt`` over it dies in
    ``_commit_native_mount`` -> ``_native_mount_ids`` with "graft N has no
    native_node_id".  The lived installer does not hit this because
    ``_install_lived_nodes`` ends by calling ``repo._native_sync_node(index)``
    for every graft.  This arm does the same, through the same call, so the
    fresh graft is registered with the native store exactly the way an
    installed one is — not a workaround, the missing half of the deposit.

    THE CAPTURE-TIME QUERY POSITION IS AN EXPLICIT VARIABLE, and it has to be.
    MEASURED here, isolated, with a receipt: ``deposit(text)`` is deterministic,
    and re-depositing the installed node's own text reproduces the installed
    payload BIT-FOR-BIT — but only while ``self_attn.live_shift`` is what the
    LIVED INSTALLER left it at.  The installer never sets it, so it is ``None``,
    and ``GptOssAttentionTC.__call__`` falls back to ``graft_seats`` (0): the
    harvest forward rotates its QUERIES at absolute positions [0, L).  After
    any serve, ``live_shift`` is ``n_sink + arena_width`` on every layer, so the
    identical ``deposit`` call rotates its queries at [live_shift, live_shift+L)
    and produces a DIFFERENT graft — layer 0 identical, every later layer
    changed, exactly as a query-position change through the attention stack
    predicts.  ``ArenaCache.deposit`` never pins the value either way, so which
    graft you get depends on whether a turn has been served yet.

    That is not a nuisance to be normalised away: it is the H-CAPTURE axis
    itself, in the geometry the mount is actually READ in.  So the capture
    position is passed in explicitly, both settings are run as named arms, and
    the value in force is on the receipt.
    """
    from scripts.grm_e2e_session import HARMONY_SINK

    # THE CAPTURE-TIME QUERY POSITION, pinned for the duration of the capture
    # and restored after.  LIVED = what the installer left (never set, so the
    # attention falls back to graft_seats == 0 and the harvest queries sit at
    # positions [0, L)).  READ = the live_shift a served turn leaves, which is
    # the geometry a MOUNT is actually read in.
    lived_capture_shift = None
    read_capture_shift = int(arena.live_shift)
    capture_shift = (
        read_capture_shift if arm in READ_GEOMETRY_ARMS else lived_capture_shift)
    before_shift = [
        getattr(layer.self_attn, "live_shift", None) for layer in arena.m.layers]
    for layer in arena.m.layers:
        layer.self_attn.live_shift = capture_shift

    fresh: list[int] = []
    detail: list[dict[str, Any]] = []
    try:
        for graft_id in [int(v) for v in mount_ids]:
            registered = capture_text_for(
                registration, probe_id, graft_id, arm)
            if registered is not None:
                text = str(registered["text"])
                source = str(registered["source"])
                node_id = registered.get("node_id")
            else:
                text = str(arena.grafts[graft_id].get("text", ""))
                source = "graft_own_text (fit-time split child)"
                node_id = None
                if not text:
                    raise RS2Error(
                        f"{probe_id}/{arm}: graft {graft_id} carries no text "
                        "to re-capture from")
            installed = payload_digest(arena.grafts[graft_id])
            if arm == "B1b":
                seam = _deposit_with_prefix(arena, HARMONY_SINK, text)
                new_id = int(seam["graft_id"])
            else:
                new_id = int(arena.deposit(text))
                seam = {
                    "graft_id": new_id,
                    "call": "ArenaCache.deposit(text)",
                    "seam": "none needed; deposit is production",
                }
            recaptured = payload_digest(arena.grafts[new_id])
            # Production's own registration of a new graft with the native
            # store — the call _install_lived_nodes makes for every node.
            native_id = repo._native_sync_node(new_id)
            arena._bump_cuda_gqa_epoch()
            fresh.append(new_id)
            detail.append({
                "installed_graft_id": int(graft_id),
                "fresh_graft_id": new_id,
                "fresh_native_node_id": (
                    None if native_id is None else int(native_id)),
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
                "capture_seam": seam,
            })
    finally:
        for layer, value in zip(arena.m.layers, before_shift):
            layer.self_attn.live_shift = value
    return {
        "arm": arm,
        "fresh_graft_ids": fresh,
        "per_node": detail,
        "capture_query_position": {
            "live_shift_pinned_for_capture": capture_shift,
            "geometry": (
                "READ (the live_shift a served turn leaves; the geometry a "
                "MOUNT is actually read in)"
                if arm in READ_GEOMETRY_ARMS else
                "LIVED (what grm_det1_3_gpu._install_lived_nodes left: never "
                "set, so GptOssAttentionTC falls back to graft_seats == 0 and "
                "the harvest queries sit at positions [0, L))"),
            "lived_capture_shift": lived_capture_shift,
            "read_capture_shift": read_capture_shift,
            "restored_to": [
                (None if v is None else int(v)) for v in before_shift[:1]],
            "why": (
                "ArenaCache.deposit does not pin the capture-time query "
                "position, so which graft a deposit produces depends on "
                "whether a turn has been served yet. RS2 makes it an explicit "
                "arm variable instead of an accident."),
        },
        "all_payloads_identical": bool(
            detail and all(row["payload_identical"] for row in detail)),
    }


def _row(probe: Mapping[str, Any], probe_id: str, session_id: str, arm: str,
         variant: str, registered: bool, question: str, asked: str,
         answer: str, mounts: Sequence[int],
         picks_requested: Sequence[int] | None,
         feed_receipt: Mapping[str, Any] | None, mass: Mapping[str, Any],
         info: Mapping[str, Any], mount_n: int, started: int,
         ) -> dict[str, Any]:
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
        "question": question,
        "question_asked": asked,
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


def run_arm(session_id: str, arm: str) -> dict[str, Any]:
    """One arm over ONE fixture: build, then serve the fixture's probe plan."""
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual)

    if arm not in ARMS:
        raise RS2Error(f"unknown arm {arm!r}; registered arms are {ARMS}")
    registration = read_registration()
    targets = set(probes_for_session(registration, session_id))
    if not targets:
        raise RS2Error(f"no registered probe belongs to fixture {session_id}")
    plan_items = [
        item for item in arm_plan(registration, session_id)
        if item["arm"] == arm]

    frame = json.loads(RUNTIME_FRAME.read_text(encoding="utf-8"))
    flags = frame["resolved_flags"]
    fixture = json.loads(
        (SUP_FIXTURES / f"{session_id}.json").read_text(encoding="utf-8"))
    plan = _probe_plan(session_id)

    # The lived switches, pinned for this process.  Fixes ON (production),
    # demand OFF (the order: "fixes ON, demand OFF"), spec frame (the escape is
    # left UNSET so the fail-closed default selects the ephemeral boat).
    os.environ["GRM_LSR_FIXES"] = "1"
    os.environ["GRM_DEMAND_NGH"] = "0"
    os.environ.pop(PERSISTENT_BOAT_ENV, None)

    repo_dir = Path(tempfile.mkdtemp(prefix=f"grm_rs2_{session_id}_{arm}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    model_info: Any = None
    node_to_idx: dict[str, int] = {}
    live_at_install: list[int] = []
    frame_ephemeral = True
    storage_bits_observed: Any = None
    layer_inventory: dict[str, Any] = {}
    try:
        e2e, model, tokenizer, repo, model_info = _load_lived_repo(
            repo_dir, frame, storage_bits="lived")
        arena = repo.arena
        frame_ephemeral = bool(getattr(arena, "ephemeral", False))
        storage_bits_observed = getattr(arena, "storage_bits", None)
        if storage_bits_observed != int(flags["graft_storage_bits"]):
            raise RS2Error(
                f"{arm} must run on the LIVED storage payload "
                f"({flags['graft_storage_bits']} bits) but the arena reports "
                f"storage_bits={storage_bits_observed!r}")
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
                # through the B0 path, because the state evolution the
                # registered probes inherit has to match B0's.
                started = time.time_ns()
                answer, info, mounts, mass = _serve_ladder(
                    repo, e2e, question, flags)
                info["rs2_arm_note"] = (
                    "not a registered probe: served through the B0 path so the "
                    "repository state this arm's registered probes inherit "
                    "matches B0's")
                served.append(_row(
                    probe, probe_id, session_id, arm, "registered", False,
                    question, question, answer, mounts, None, None, mass, info,
                    int(arena.cur_mount_n), started))
                continue

            for item in items:
                started = time.time_ns()
                variant = str(item["variant"])
                mount_ids = [int(v) for v in item["mount_ids"]]
                feed_receipt: dict[str, Any] | None = None
                picks_requested: list[int] | None = None

                if arm == "B0" and variant == "registered":
                    answer, info, mounts, mass = _serve_ladder(
                        repo, e2e, question, flags)
                    served.append(_row(
                        probe, probe_id, session_id, arm, variant, True,
                        question, question, answer, mounts, None, None, mass,
                        info, int(arena.cur_mount_n), started))
                    continue

                # Every other (arm, variant) needs the B0 PRELUDE first.
                prelude = _prelude(repo, e2e, question, flags, arena, mount_ids)
                base = _snapshot_counterfactual(arena)
                native_base = _snapshot_native_ids(repo)
                missing = [v for v in mount_ids if int(v) >= len(arena.grafts)]
                if missing:
                    raise RS2Error(
                        f"{probe_id}/{arm}/{variant}: mount ids {missing} do "
                        f"not exist after the B0 prelude "
                        f"({len(arena.grafts)} grafts)")
                _clear_boat(arena)

                if arm == "B0":
                    # B0's solace-fact variant: production routing decides B0's
                    # mount, so the variant is the explicit-mount serve over the
                    # registered fact node with everything else B0's.
                    picks_requested = list(mount_ids)
                    answer, info, mounts, mass = _serve_direct(
                        repo, e2e, question, flags, picks_requested)
                elif arm == "B5":
                    feed_receipt = _feed_live(
                        repo, e2e, fixture,
                        [str(v) for v in item["live_node_ids"]])
                    picks_requested = []
                    answer, info, mounts, mass = _serve_direct(
                        repo, e2e, question, flags, picks_requested)
                elif arm in CAPTURE_ARMS:
                    capture = _capture_and_mount(
                        repo, arena, registration, probe_id, arm, mount_ids)
                    picks_requested = [
                        int(v) for v in capture["fresh_graft_ids"]]
                    answer, info, mounts, mass = _serve_direct(
                        repo, e2e, question, flags, picks_requested)
                    info["rs2_capture"] = capture
                else:  # B3a / B3b
                    picks_requested = list(mount_ids)
                    mount_ntok = sum(
                        int(arena.grafts[i]["ntok"]) for i in picks_requested)
                    mount_pos0 = (
                        int(arena.live_shift) - int(mount_ntok)
                        if arm == "B3a" else int(arena.n_sink))
                    answer, info, mounts, mass = _serve_positioned(
                        repo, e2e, question, flags, picks_requested,
                        mount_pos0)

                info["rs2_b0_prelude"] = prelude
                served.append(_row(
                    probe, probe_id, session_id, arm, variant, True, question,
                    question, answer, mounts, picks_requested, feed_receipt,
                    mass, info, int(info.get("rs2_cur_mount_n_at_serve", 0)),
                    started))
                # Leave the arena where the arm found it, so the NEXT probe in
                # the plan inherits B0's state evolution rather than this arm's.
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
        "phase": "RS2",
        "order": "orders/GRM_RS2_MOUNT_READ_DEFICIT.md",
        "arm": arm,
        "session_id": session_id,
        "registered_probe_ids": sorted(targets),
        "frame": {
            "ephemeral": bool(frame_ephemeral),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
            "spec": (
                "GRM-EB1: the chat log is not kept in memory context; any chat "
                "recall on facts is pulled via GRM"),
        },
        "switches": {
            "GRM_LSR_FIXES": os.environ.get("GRM_LSR_FIXES"),
            "GRM_DEMAND_NGH": os.environ.get("GRM_DEMAND_NGH"),
            "storage_bits": storage_bits_observed,
        },
        "arena_width": int(flags["arena_width"]),
        "layer_inventory": layer_inventory,
        "deposit_protocol": (
            "CHRONOLOGICAL_ARENA_FEED (grm_det1_3_gpu._install_lived_nodes), "
            "the LIVED arm; under the SPEC frame that call reaches "
            "ArenaCache.deposit(text) through feed's ephemeral branch"),
        "fixture_node_to_idx": {str(k): int(v) for k, v in node_to_idx.items()},
        "live_graft_ids_after_install": live_at_install,
        "probes": served,
        "registration": file_record(REGISTRATION),
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_rs2_mount_read_gpu": file_record(Path(__file__).resolve()),
            "grm_rs2_registration": file_record(
                ROOT / "scripts" / "grm_rs2_registration.py"),
            "grm_rs1_read_strength_gpu": file_record(
                ROOT / "scripts" / "grm_rs1_read_strength_gpu.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "gpt_oss20b_tc": file_record(ROOT / "core" / "gpt_oss20b_tc.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
        },
    }


# ======================================================================
# Table assembly (pure — CPU tested)
# ======================================================================
def assemble_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The G3 table: one row per (probe, variant, arm), in that order.

    The columns are exactly the ones the order names: served, correct, mounted
    mass on FULL layers, mounted mass on SLIDING layers, live mass, sink mass,
    top layer.  Pure shaping over already-measured rows.
    """
    order = {arm: index for index, arm in enumerate(ARMS)}
    out: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda r: (str(r["probe_id"]), str(r.get("variant", "registered")),
                       order.get(str(r["arm"]), 99)),
    ):
        mass = dict(row.get("mass") or {})
        by_type = dict(mass.get("by_layer_type") or {})
        full = dict(by_type.get(FULL) or {})
        sliding = dict(by_type.get(SLIDING) or {})
        full_mean = dict(full.get("mean_over_answer_positions") or {})
        sliding_mean = dict(sliding.get("mean_over_answer_positions") or {})
        top_full = dict(full.get("top_contributing_layer") or {})
        top_sliding = dict(sliding.get("top_contributing_layer") or {})
        out.append({
            "probe_id": str(row["probe_id"]),
            "variant": str(row.get("variant", "registered")),
            "arm": str(row["arm"]),
            "served": str(row["served_answer"]),
            "correct": bool(row["correct"]),
            "mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
            "mounted_mass_full_layers": full_mean.get("mounted_mass"),
            "mounted_mass_sliding_layers": sliding_mean.get("mounted_mass"),
            "live_mass_full_layers": full_mean.get("live_mass"),
            "live_mass_sliding_layers": sliding_mean.get("live_mass"),
            "sink_mass_full_layers": full_mean.get("physical_sink_mass"),
            "sink_mass_sliding_layers": sliding_mean.get("physical_sink_mass"),
            "learned_sink_mass_full_layers": full_mean.get("learned_sink_mass"),
            "top_layer_full": top_full.get("layer_ordinal"),
            "top_layer_full_mounted_mass": top_full.get("mounted_mass"),
            "top_layer_sliding": top_sliding.get("layer_ordinal"),
            "top_layer_sliding_mounted_mass": top_sliding.get("mounted_mass"),
            "sliding_window_reaches_mount": (
                dict(mass.get("sliding_window_reach") or {})
                .get("always_reaches_mount")),
        })
    return out


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
        raise RS2Error("--session-id is required for the arm command")
    if not args.arm:
        raise RS2Error("--arm is required for the arm command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_arm(str(args.session_id), str(args.arm))
    path = emit(payload, f"grm_rs2_{args.arm}_{args.session_id}")
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
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
