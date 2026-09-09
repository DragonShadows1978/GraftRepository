"""GRM Stage C — the production D-NGH demand detector.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md``.

WHAT THIS IS.  ``GRM-DET1`` raced four "the model needs a memory it does not
have" detectors on ten planted-miss / served-control pairs.  D-NGH won
(recall 1.00, FP 0.10, F1 0.952); D-LQR was REFUTED-STRUCTURAL, so live-query
routing is NOT a fetch source anywhere in this file.  The race observer,
``scripts/grm_det1_common.DetectorObserver``, is FROZEN instrumentation: it
lives under ``scripts/`` and it also carries D-LQR and D-ENT hooks that
production has no business paying for.  This module is the PRODUCTION D-NGH
observer — the same capture, the same score, the same decision rule, usable
inside ``ArenaCache.step`` and ``grm_e2e_session._probe_ladder_chat``.

THE SCORE (carried, not re-derived).  Per generated token, over GPT-OSS
FULL-ATTENTION layers only (sliding layers are suppressed exactly as the race
suppressed them), recompute the last query row's attention distribution over
[physical sink | mounted band | live tail | learned sink] and take the
head-mean probability mass on the MOUNTED band.  ``mounted_mass`` for a token
is the mean of that quantity across the full-attention layers.

THE RULE (carried, not refit).  Fire on the FIRST token whose
``mounted_mass`` is STRICTLY BELOW the registered threshold
(``config/grm_demand_registered.json``: 0.3380523274342219, fit on TWO
calibration turns — a thin envelope, and every receipt says so).  Equality
does NOT fire.

REFUSAL IS LOUD.  A non-GPT-OSS arena, or a GPT-OSS arena whose full layers
are not on the standard attention path, sets ``demand_supported=False`` with
a reason.  It never silently no-ops into "no demand detected", because that
is indistinguishable from "the memory was there".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTERED_CONFIG_PATH = ROOT / "config" / "grm_demand_registered.json"

#: The env switch. DEFAULT OFF; an unknown token fails CLOSED to OFF.
ENV_NAME = "GRM_DEMAND_NGH"
_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off"))

#: GRM-SC2: the early-abort switch. DEFAULT **ON whenever demand is ON**.
#:
#: The direction is deliberately the opposite of ``GRM_DEMAND_NGH``'s.  That
#: switch fails CLOSED to OFF because turning the demand loop on is a change to
#: what the stack SERVES, and a typo must never make that change.  This switch
#: cannot change what the stack serves at all: a fired turn either serves the
#: grounded demand trip (identical either way) or resumes the suspended attempt
#: and serves exactly the text the un-aborted attempt would have served.  What
#: it changes is only how many tokens are generated on the way there.  So an
#: unreadable token fails to the DEFAULT (on), and only an explicit false token
#: selects the SC1 generate-the-whole-wrong-answer path.
ENV_EARLY_ABORT = "GRM_DEMAND_EARLY_ABORT"

#: Registered cap: at most ONE demand trip per turn, ever. A second fire on
#: the demand trip is RECORDED and NOT acted on.
DEMAND_TRIP_CAP = 1

#: The only fetch source SC1 registers. NOT D-LQR live-query fingerprints
#: (REFUTED-STRUCTURAL in the race) — the model's own partial output is
#: appended to the question and the existing route()/admission path is
#: re-run over it.
DEMAND_FETCH_SOURCE = "question_plus_prefix_reroute"

#: ``info`` keys this module writes. All share the ``demand_`` prefix so the
#: P2B route receipt persists them via ROUTE_RECEIPT_INFO_PREFIXES.
DEMAND_INFO_PREFIX = "demand_"


class DemandError(RuntimeError):
    """The production demand detector could not run as registered."""


class DemandAbort(BaseException):
    """GRM-SC2: D-NGH fired mid-generation; stop generating THIS attempt.

    Derived from ``BaseException``, not ``Exception``, on purpose.  It is
    raised from inside a wrapped ``arena._forward`` and has to travel up
    through the generation loop untouched.  Anything on that path that catches
    broad ``Exception`` to convert a decode failure into a fallback would
    otherwise swallow the abort and turn a control-flow signal into a silent
    behaviour change.  ``_attempt`` catches this class by name, and nothing
    else does.

    ``token_index`` is the answer position that fired -- the FIRST position
    strictly below the registered line.  The tokens generated BEFORE it are the
    ones the model produced while still reading its mounts; they are kept, both
    as the demand query's prefix and as the resume point.
    """

    def __init__(self, token_index: int, mounted_mass: float) -> None:
        super().__init__(
            f"D-NGH fired at answer token {int(token_index)} "
            f"(mounted_mass={float(mounted_mass)!r})")
        self.token_index = int(token_index)
        self.mounted_mass = float(mounted_mass)


# --------------------------------------------------------------------------
# Registered constants
# --------------------------------------------------------------------------


def load_registered(path: Path | None = None) -> dict[str, Any]:
    """Read the registered D-NGH constant and its provenance.

    Fails closed: a config that does not declare ``carried_not_refit`` or
    that names a different score/direction is refused rather than used, so a
    silently-edited constant cannot become a live threshold.
    """
    target = Path(REGISTERED_CONFIG_PATH if path is None else path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemandError(
            f"registered demand config is unreadable: {target}: {exc}") from exc
    if payload.get("schema") != "grm.demand.registered.v1":
        raise DemandError(
            f"unexpected demand config schema: {payload.get('schema')!r}")
    if payload.get("detector") != "D-NGH":
        raise DemandError(
            f"demand config is not D-NGH: {payload.get('detector')!r}")
    if payload.get("carried_not_refit") is not True:
        raise DemandError(
            "demand config does not declare the threshold CARRIED from the "
            "race; SC1 registers no new number")
    if payload.get("score") != "per-token full-layer mean mounted_mass":
        raise DemandError(f"demand score drifted: {payload.get('score')!r}")
    if payload.get("direction") != "trigger_if_strictly_below":
        raise DemandError(
            f"demand direction drifted: {payload.get('direction')!r}")
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold):
        raise DemandError(f"demand threshold is not finite: {threshold!r}")
    return payload


def registered_threshold(path: Path | None = None) -> float:
    """The line production actually fires on.

    GRM-SC2 recorded a wider-calibration ``candidate_threshold`` in the same
    config file.  Production does NOT read it.  It reads ``threshold`` -- the
    value carried verbatim from the race -- and keeps reading it until a config
    sets ``adopted_by`` to name who adopted the candidate and under which
    order.  SC2 does not set that field: the candidate is a REPORTED number for
    David to rule on, not a live one.

    A config that sets ``adopted_by`` without carrying a ``candidate_threshold``
    to adopt is refused rather than silently falling back, because that shape is
    indistinguishable from a half-applied edit.
    """
    payload = load_registered(path)
    adopted_by = payload.get("adopted_by")
    if adopted_by in (None, "", False):
        return float(payload["threshold"])
    if not isinstance(adopted_by, str) or not adopted_by.strip():
        raise DemandError(
            f"demand config `adopted_by` must name an adopter: {adopted_by!r}")
    if "candidate_threshold" not in payload:
        raise DemandError(
            "demand config sets `adopted_by` but carries no "
            "`candidate_threshold` to adopt")
    candidate = float(payload["candidate_threshold"])
    if not np.isfinite(candidate):
        raise DemandError(f"candidate threshold is not finite: {candidate!r}")
    return candidate


def demand_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve ``GRM_DEMAND_NGH``: explicit caller, then env, then OFF.

    DEFAULT OFF, and an unrecognised token fails CLOSED to OFF — the opposite
    direction from ``GRM_LSR_FIXES`` on purpose: this mechanism is not the
    production default until David flips it, so a typo must never turn it on.
    """
    if explicit is not None:
        return bool(explicit)
    env = os.environ if environ is None else environ
    value = str(env.get(ENV_NAME, "")).strip().casefold()
    return value in _ENV_TRUE


def early_abort_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve ``GRM_DEMAND_EARLY_ABORT``: explicit caller, then env, then ON.

    GRM-SC2 registers this DEFAULT ON whenever the demand loop is on.  Only an
    explicit false token (``0``/``false``/``no``/``off``) selects the SC1 path
    that generates the whole wrong answer before taking the trip; an unknown
    or unset token resolves to the default, because this switch cannot change
    what the turn serves -- only how much is generated getting there.

    The caller is responsible for not consulting this at all when demand is
    OFF; with demand off there is no detector to fire and nothing to abort.
    """
    if explicit is not None:
        return bool(explicit)
    env = os.environ if environ is None else environ
    value = str(env.get(ENV_EARLY_ABORT, "")).strip().casefold()
    return value not in _ENV_FALSE


# --------------------------------------------------------------------------
# Support / refusal
# --------------------------------------------------------------------------


def arena_support(arena: Any) -> dict[str, Any]:
    """Can this arena carry the production D-NGH observer?

    LOUD, never silent.  The returned mapping always has ``demand_supported``
    and, when False, ``demand_unsupported_reason``.
    """
    layers = list(getattr(getattr(arena, "m", None), "layers", ()) or ())
    if not layers:
        return {
            "demand_supported": False,
            "demand_unsupported_reason": "arena has no model layers",
        }
    try:
        full = [
            layer for layer in layers
            if str(getattr(layer.self_attn, "layer_type", "")) == "full_attention"
        ]
    except AttributeError:
        return {
            "demand_supported": False,
            "demand_unsupported_reason":
                "arena layers do not expose self_attn.layer_type; the D-NGH "
                "observer supports the GPT-OSS attention path only",
        }
    if not full:
        return {
            "demand_supported": False,
            "demand_unsupported_reason":
                "no GPT-OSS full-attention layers found; D-NGH is defined on "
                "the full-attention band only",
        }
    bad = [
        int(getattr(layer.self_attn, "layer_idx", -1)) for layer in full
        if str(getattr(layer.self_attn, "attention_mode", "")) != "standard"
    ]
    if bad:
        return {
            "demand_supported": False,
            "demand_unsupported_reason":
                "D-NGH supports the current standard GPT-OSS path only; "
                f"non-standard full layers: {sorted(bad)}",
        }
    return {"demand_supported": True, "demand_full_attention_layers": len(full)}


# --------------------------------------------------------------------------
# The observer
# --------------------------------------------------------------------------


@dataclass
class DemandObserver:
    """Read-only production D-NGH observer over ONE generation attempt.

    Byte-for-byte the same capture as the frozen race observer
    (``scripts/grm_det1_common.DetectorObserver`` with
    ``active_detectors={"D-NGH"}``): it wraps ``gpt_oss20b_tc
    .sink_attention_tc`` to accumulate per-full-layer mass rows, wraps
    ``sliding_sink_attention_tc`` to SUPPRESS sliding layers, and wraps
    ``arena._forward`` to close one token's rows into a ``mounted_mass``.
    G2 pins that equality on real GPU numbers.

    The one production difference is scope: no D-LQR capture (REFUTED) and no
    D-ENT logits work, so a served turn pays for the mass rows and nothing
    else.
    """

    arena: Any
    ngen: int
    threshold: float
    records: list[dict[str, Any]] = field(default_factory=list)
    #: GRM-SC2. When True the observer RAISES ``DemandAbort`` on the first
    #: token strictly below the line instead of letting generation run on.
    #: The row for that token is appended BEFORE the raise, so the decision
    #: the caller reads afterwards is computed from exactly the rows the
    #: un-aborted attempt would have produced up to and including the fire.
    early_abort: bool = False

    def __post_init__(self) -> None:
        self.threshold = float(self.threshold)
        self.ngen = int(self.ngen)
        self.early_abort = bool(self.early_abort)
        self.aborted_at: int | None = None
        self.records = []
        self._gpt = None
        self._original_forward = None
        self._original_sink = None
        self._original_sliding = None
        self._inside_forward = False
        self._suppress_sink = False
        self._mass_rows: list[dict[str, float]] = []
        self.expected_full_layers = 0

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "DemandObserver":
        from core import gpt_oss20b_tc as gpt

        support = arena_support(self.arena)
        if not support.get("demand_supported"):
            raise DemandError(str(support.get("demand_unsupported_reason")))
        self.expected_full_layers = int(support["demand_full_attention_layers"])

        self._gpt = gpt
        self._original_sink = gpt.sink_attention_tc
        self._original_sliding = gpt.sliding_sink_attention_tc
        self._original_forward = self.arena._forward
        observer = self

        def sink_wrapper(query, key, value, sinks, *, scale,
                         attention_mask=None, num_heads_per_kv=1):
            result = observer._original_sink(
                query, key, value, sinks,
                scale=scale,
                attention_mask=attention_mask,
                num_heads_per_kv=num_heads_per_kv,
            )
            if observer._inside_forward and not observer._suppress_sink:
                observer._mass_rows.append(observer._full_mass(
                    query, key, sinks,
                    scale=float(scale),
                    attention_mask=attention_mask,
                    num_heads_per_kv=int(num_heads_per_kv),
                ))
            return result

        def sliding_wrapper(query, key, value, sinks, *, scale,
                            sliding_window, num_heads_per_kv=1, attn_block=128,
                            allowed_mask=None):
            previous = observer._suppress_sink
            observer._suppress_sink = True
            try:
                return observer._original_sliding(
                    query, key, value, sinks,
                    scale=scale,
                    sliding_window=sliding_window,
                    num_heads_per_kv=num_heads_per_kv,
                    attn_block=attn_block,
                    allowed_mask=allowed_mask,
                )
            finally:
                observer._suppress_sink = previous

        def forward_wrapper(ids, last_only=True):
            observer._mass_rows = []
            observer._inside_forward = True
            try:
                logits = observer._original_forward(ids, last_only=last_only)
            finally:
                observer._inside_forward = False
            mass = observer._summarize_mass(observer._mass_rows)
            index = len(observer.records)
            observer.records.append({
                "token_index": index,
                # The greedy prediction at this answer position. Captured so
                # a fired turn can rebuild the model's own partial output up
                # to the fire index WITHOUT re-tokenizing the decoded string
                # (decode->encode is not a round trip on BPE, and the demand
                # query must be the exact prefix the detector fired on).
                "prediction_token_id": int(np.asarray(logits).argmax()),
                "mounted_mass": float(mass["mounted_mass"]),
                "physical_sink_mass": float(mass["physical_sink_mass"]),
                "live_mass": float(mass["live_mass"]),
                "learned_sink_mass": float(mass["learned_sink_mass"]),
                "full_attention_layers": int(observer.expected_full_layers),
            })
            # GRM-SC2 EARLY ABORT. The carried rule, applied the instant it can
            # be applied instead of after the whole answer is generated: FIRST
            # token STRICTLY BELOW the line. Equality still does not fire.
            #
            # The row above is already appended, so `finish()`/`decide()` see
            # the same rows the un-aborted attempt would have produced through
            # this position -- the decision is identical, only the tokens after
            # it are never generated. The raise happens AFTER the real logits
            # were computed, so the arena's KV cache is in exactly the state a
            # non-aborting forward would have left it in: that is the resume
            # point, and it is why resuming is byte-identical rather than
            # approximately identical.
            # Prior art: GRM-SC2 / SC1 flush-drop convention (project, 2026).
            # Keep the unused final prediction as raw evidence, but only
            # answer positions may suspend. This boundary repair is new;
            # the detector rule and completed-answer decision are unchanged.
            if (observer.early_abort and index < observer.ngen
                    and observer.aborted_at is None
                    and float(mass["mounted_mass"]) < observer.threshold):
                observer.aborted_at = int(index)
                raise DemandAbort(index, float(mass["mounted_mass"]))
            return logits

        gpt.sink_attention_tc = sink_wrapper
        gpt.sliding_sink_attention_tc = sliding_wrapper
        self.arena._forward = forward_wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_forward is not None:
            self.arena._forward = self._original_forward
        if self._gpt is not None:
            self._gpt.sink_attention_tc = self._original_sink
            self._gpt.sliding_sink_attention_tc = self._original_sliding
        return False

    # -- the score ------------------------------------------------------
    def _full_mass(
        self,
        query,
        key,
        sinks,
        *,
        scale: float,
        attention_mask,
        num_heads_per_kv: int,
    ) -> dict[str, float]:
        """The frozen race computation, operand for operand.

        Identical to ``DetectorObserver._full_mass``; any divergence here is
        exactly what G2's bit-identity gate catches.
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
        sink_logits = sinks.reshape([1, heads, 1, 1]).expand([batch, heads, 1, 1])
        probs = gpt.tc.cat([scores, sink_logits], dim=-1).softmax(-1)
        values = np.asarray(probs.float().numpy(), dtype=np.float32)[0, :, 0, :]
        start = int(self.arena.n_sink)
        end = min(key_rows, start + int(self.arena.cur_mount_n))
        physical_sink = values[:, :min(start, key_rows)].sum(axis=1)
        mounted = values[:, start:end].sum(axis=1) if end > start else np.zeros(heads)
        live = values[:, end:key_rows].sum(axis=1)
        learned = values[:, key_rows]
        return {
            "physical_sink_mass": float(np.mean(physical_sink)),
            "mounted_mass": float(np.mean(mounted)),
            "live_mass": float(np.mean(live)),
            "learned_sink_mass": float(np.mean(learned)),
        }

    def _summarize_mass(
        self, rows: Sequence[Mapping[str, float]],
    ) -> dict[str, float]:
        if len(rows) != self.expected_full_layers:
            raise DemandError(
                "D-NGH full-layer capture count mismatch: expected "
                f"{self.expected_full_layers}, observed {len(rows)}")
        mass = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "physical_sink_mass", "mounted_mass", "live_mass",
                "learned_sink_mass",
            )
        }
        total = sum(mass.values())
        if not (0.98 <= total <= 1.02):
            raise DemandError(
                f"D-NGH mass partition does not sum to one: {mass}")
        return mass

    # -- the decision ---------------------------------------------------
    def finish(self) -> list[dict[str, Any]]:
        """The answer-position rows, with the race's flush-drop convention.

        ``ArenaCache._attempt`` runs ONE extra forward past ``ngen`` solely to
        commit the final predicted token to KV; its logits are unused and it
        is not an answer position.  The race dropped it and so does this.
        """
        kept = list(self.records)
        # GRM-SC2: an ABORTED attempt never reached the final commit forward,
        # so there is no unused flush row to drop. Dropping one here would
        # silently discard the FIRE row itself whenever the fire landed on the
        # last captured position, which would turn a fired turn into an
        # un-fired one. The convention applies to completed attempts only.
        if self.aborted_at is None and len(kept) == self.ngen + 1:
            kept = kept[:-1]
        if len(kept) > self.ngen:
            raise DemandError(
                f"observer captured {len(kept)} answer rows for ngen={self.ngen}")
        for index, row in enumerate(kept):
            row["token_index"] = int(index)
        return kept

    def decision(self) -> dict[str, Any]:
        return decide(self.finish(), self.threshold)


def decide(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """The carried D-NGH rule: FIRST token STRICTLY BELOW ``threshold``.

    Equality does NOT fire — the race's ``trigger_if_strictly_below``,
    reproduced here rather than reinterpreted.  ``demand_min_mass`` is the
    minimum over every answer position, fired or not, so an un-fired turn
    still reports how close it came.
    """
    threshold = float(threshold)
    fired = False
    index: int | None = None
    for row in rows:
        if float(row["mounted_mass"]) < threshold:
            fired = True
            index = int(row["token_index"])
            break
    minimum = (
        min(float(row["mounted_mass"]) for row in rows) if rows else None)
    return {
        "demand_fired": bool(fired),
        "demand_token_index": index,
        "demand_min_mass": minimum,
        "demand_threshold": threshold,
        "demand_token_count": len(rows),
    }


def demand_prefix_text(arena: Any, rows: Sequence[Mapping[str, Any]],
                       token_index: int) -> str:
    """The model's own partial output UP TO (not including) the fire token.

    The detector fires at the first answer position whose mounted mass fell
    below the envelope; everything the model emitted BEFORE that position is
    the part it produced while still reading its mounts, so that is the part
    worth using as query augmentation.  The fire token itself is the first
    one the model produced while adrift and is deliberately excluded.
    """
    index = int(token_index)
    ids = [int(row["prediction_token_id"]) for row in rows[:index]]
    if not ids:
        return ""
    return str(arena.decode(ids))


def demand_query_text(question: str, prefix: str) -> str:
    """Question PLUS the generated prefix — the registered fetch source.

    Deliberately NOT D-LQR: no live-query fingerprints, no new scoring law.
    The augmented text goes through the SAME ``route()``/admission path the
    turn already used, so the demand trip inherits every routing law the turn
    was already subject to.
    """
    prefix = str(prefix).strip()
    if not prefix:
        return str(question)
    return f"{question}\n{prefix}"


def demand_query_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def unsupported_info(reason: str) -> dict[str, Any]:
    """The loud refusal payload, in ``demand_``-prefixed ``info`` form."""
    return {
        "demand_supported": False,
        "demand_unsupported_reason": str(reason),
        "demand_fired": False,
        "demand_token_index": None,
        "demand_min_mass": None,
        "demand_served": "original",
    }


def demand_info_fields(
    *,
    supported: bool,
    decision: Mapping[str, Any] | None = None,
    served: str = "original",
    query_text_sha256: str | None = None,
    ranking: Sequence[int] | None = None,
    fetched: Sequence[int] | None = None,
    refired: bool | None = None,
    refire_token_index: int | None = None,
    trip_taken: bool = False,
    trip_grounded: bool | None = None,
    unsupported_reason: str | None = None,
    threshold: float | None = None,
    prefix_token_count: int | None = None,
    early_abort: bool | None = None,
    abort_token_index: int | None = None,
    tokens_generated_before_abort: int | None = None,
    tokens_saved: int | None = None,
    resumed_original: bool | None = None,
    wall_ms_attempt: float | None = None,
    wall_ms_trip: float | None = None,
    wall_ms_resume: float | None = None,
) -> dict[str, Any]:
    """Assemble the registered ``demand_*`` receipt block.

    EVERY field is ``demand_``-prefixed, so ``core/grm_three_pass
    .ROUTE_RECEIPT_INFO_PREFIXES`` persists the whole block with the one-line
    tuple addition SC1 is authorized to make and no other change to that file.
    """
    out: dict[str, Any] = {
        "demand_supported": bool(supported),
        "demand_fetch_source": DEMAND_FETCH_SOURCE,
        "demand_trip_cap": int(DEMAND_TRIP_CAP),
        "demand_served": str(served),
        "demand_trip_taken": bool(trip_taken),
    }
    if unsupported_reason is not None:
        out["demand_unsupported_reason"] = str(unsupported_reason)
    if threshold is not None:
        out["demand_threshold"] = float(threshold)
    if decision is not None:
        out["demand_fired"] = bool(decision.get("demand_fired", False))
        token_index = decision.get("demand_token_index")
        out["demand_token_index"] = (
            None if token_index is None else int(token_index))
        minimum = decision.get("demand_min_mass")
        out["demand_min_mass"] = None if minimum is None else float(minimum)
        out["demand_token_count"] = int(decision.get("demand_token_count", 0))
        if decision.get("demand_threshold") is not None:
            out["demand_threshold"] = float(decision["demand_threshold"])
    else:
        out.setdefault("demand_fired", False)
        out.setdefault("demand_token_index", None)
        out.setdefault("demand_min_mass", None)
    if query_text_sha256 is not None:
        out["demand_query_text_sha256"] = str(query_text_sha256)
    if prefix_token_count is not None:
        out["demand_prefix_token_count"] = int(prefix_token_count)
    if ranking is not None:
        out["demand_ranking"] = [int(v) for v in ranking]
    if fetched is not None:
        out["demand_fetched"] = [int(v) for v in fetched]
    if refired is not None:
        out["demand_refired"] = bool(refired)
        out["demand_refire_acted_on"] = False
        out["demand_refire_token_index"] = (
            None if refire_token_index is None else int(refire_token_index))
    if trip_grounded is not None:
        out["demand_trip_grounded"] = bool(trip_grounded)
    # --- GRM-SC2 early-abort receipt block --------------------------------
    if early_abort is not None:
        out["demand_early_abort"] = bool(early_abort)
    if abort_token_index is not None:
        out["demand_abort_token_index"] = int(abort_token_index)
    if tokens_generated_before_abort is not None:
        out["demand_tokens_generated_before_abort"] = int(
            tokens_generated_before_abort)
    if tokens_saved is not None:
        out["demand_tokens_saved"] = int(tokens_saved)
    if resumed_original is not None:
        out["demand_resumed_original"] = bool(resumed_original)
    if wall_ms_attempt is not None:
        out["demand_wall_ms_attempt"] = float(wall_ms_attempt)
    if wall_ms_trip is not None:
        out["demand_wall_ms_trip"] = float(wall_ms_trip)
    if wall_ms_resume is not None:
        out["demand_wall_ms_resume"] = float(wall_ms_resume)
    return out
