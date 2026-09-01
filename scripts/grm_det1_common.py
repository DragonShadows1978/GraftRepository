#!/usr/bin/env python3
"""Shared, mostly CPU-only machinery for ORDER GRM-DET1.

The live observer is deliberately script-local.  It installs read-only hooks
only while one registered probe attempt is running and restores every touched
callable/attribute in ``finally``.  Importing this module changes no production
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORDER_PATH = ROOT / "orders" / "GRM_DET1_DEMAND_DETECTOR_RACE.md"
ARTIFACT_ROOT = ROOT / "artifacts" / "grm_det1"
VERBAL_QUESTION = "Do you have the information needed to answer? YES/NO"
DETECTORS = ("D-LQR", "D-NGH", "D-ENT", "D-VERB")
MECHANISTIC = ("D-LQR", "D-NGH", "D-ENT")

# --- GRM-DET1.11 achieved-count amendment -------------------------------
# The registered evaluation population is twelve planted/served pairs.  The
# ACHIEVED population is however many of those slots produced a lawful lived
# served control; the campaign runs at that count and refuses below the
# floor.  Both numbers live here, in the shared base module, so exactly one
# definition governs the driver, the registry, and the analyzer alike.  The
# floor is registered before the gate it decides and is never adjusted after
# seeing results.
REGISTERED_EVAL_PAIR_COUNT = 12
REGISTERED_CALIBRATION_PAIR_COUNT = 2
REGISTERED_FIXTURE_COUNT = 14
ACHIEVED_EVAL_PAIR_FLOOR = 8
CALIBRATION_IDS = ("e2e_t09_cypher_bridge", "e2e_t16_lyra_dock")
EVAL_E2E_IDS = (
    "e2e_t05_orion_pin",
    "e2e_t13_orion_pin",
    "e2e_t19_nova_key",
    "e2e_t22_mira_seal",
    "e2e_t24_terra_port",
    "e2e_t26_ember_code",
    "e2e_t30_atlas_tone",
)


class DETError(RuntimeError):
    """A frozen DET1 invariant was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any, where: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DETError(f"non-finite value at {where}: {value!r}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{where}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _finite(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(path: Path, value: Any) -> Path:
    payload = canonical_json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


def write_content_addressed(directory: Path, stem: str, value: Any) -> Path:
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    path = Path(directory) / f"{stem}_{digest[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DETError(f"content-address collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


def append_jsonl_once(path: Path, row: Mapping[str, Any], *, key: str) -> None:
    path = Path(path)
    payload = canonical_json_bytes(dict(row)).decode("utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            existing[str(value[key])] = json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False)
    compact = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)
    row_key = str(row[key])
    if row_key in existing:
        if existing[row_key] != compact:
            raise DETError(f"append-only collision for {row_key} in {path}")
        return
    _finite(row)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload.replace("\n", "") + "\n")
        handle.flush()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DETError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_record(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        shown = str(path.relative_to(root.resolve()))
    except ValueError:
        shown = str(path)
    return {"path": shown, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def source_inventory(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        file_record(path)["path"]: {
            "bytes": file_record(path)["bytes"],
            "sha256": file_record(path)["sha256"],
        }
        for path in paths
    }


# DET1.10: separator glyphs that may vary between a fixture's written value
# and a served answer.  ASCII hyphen and U+2010/U+2011 are already collapsed to
# "-" by normalize_value_text; whitespace is added here as an equivalent
# separator.  Underscore is deliberately EXCLUDED: it participates in the word
# boundary class, so treating it as a separator would let "cobalt_1_india"
# straddle a boundary it is supposed to define.
VALUE_SEPARATOR_PATTERN = r"[-\s]+"
_VALUE_SEPARATOR_RUN = re.compile(VALUE_SEPARATOR_PATTERN)


def value_separator_regex(word: str) -> str:
    """Build a token-exact, separator-agnostic pattern for a value.

    DET1.10 registers separator glyphs (ASCII hyphen, U+2010, U+2011, space)
    as presentation WITHIN a value token sequence.  The token payloads and the
    token COUNT stay load-bearing: the tokens are escaped literally and joined
    by a mandatory separator run, so "Cobalt-2-India" (wrong token) and
    "Cobalt India" (missing token) still fail against "Cobalt-1-India".
    """
    tokens = [token for token in _VALUE_SEPARATOR_RUN.split(word) if token]
    if not tokens:
        return re.escape(word)
    return VALUE_SEPARATOR_PATTERN.join(re.escape(token) for token in tokens)


def whole_word(text: str, word: str) -> bool:
    normalized_text = normalize_value_text(text)
    normalized_word = normalize_value_text(word)
    return bool(re.search(
        rf"(?<![A-Za-z0-9_-]){value_separator_regex(normalized_word)}"
        rf"(?![A-Za-z0-9_-])",
        normalized_text,
        flags=re.IGNORECASE,
    ))


def normalize_value_text(value: str) -> str:
    """Normalize DET value semantics without changing raw answer receipts.

    DET1.4 registers Markdown emphasis and U+2010/U+2011 hyphens as glyph
    presentation, not value differences.  Callers must retain the original
    answer for byte/behavior comparisons and use this projection only for
    expected/stale/wrong-value classification.
    """
    text = str(value).replace("\u2010", "-").replace("\u2011", "-")
    # Remove paired Markdown emphasis delimiters while leaving their payload
    # byte order unchanged.  Iterate so nested bold/italic is also projected.
    emphasis = re.compile(r"(\*\*|__|\*|_)([^\n]+?)\1")
    previous = None
    while previous != text:
        previous = text
        text = emphasis.sub(r"\2", text)
    return text.casefold()


def contains_value(text: str, value: str) -> bool:
    return whole_word(text, value)


def routing_index_digest(arena: Any) -> dict[str, Any]:
    """Digest every field that can affect the registered D-LQR index.

    Mounted K/V is intentionally excluded: DET1.4 verifies it through the
    lived snapshot.  This projection binds candidate membership, centroids,
    length debias inputs, text identity, and revision metadata so planted
    withholding cannot silently delete or rebuild the full detector index.
    """
    candidates = [int(value) for value in arena._route_cand_base()]

    def stable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): stable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        if isinstance(value, set):
            return sorted(stable(item) for item in value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def array_record(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            array = value
        elif hasattr(value, "numpy"):
            array = value.numpy()
        else:
            array = np.asarray(value)
        array = np.ascontiguousarray(array)
        return {
            "shape": [int(item) for item in array.shape],
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    rows = []
    for index in candidates:
        graft = arena.grafts[index]
        child_cents = graft.get("child_cents") or ()
        row = {
            "index": index,
            "native_node_id": graft.get("native_node_id"),
            "kind": str(graft.get("kind", "turn")),
            "retired": bool(graft.get("retired")),
            "ntok": int(graft.get("ntok", 0) or 0),
            "text_sha256": hashlib.sha256(
                str(graft.get("text", "") or "").encode("utf-8")
            ).hexdigest(),
            "cent": array_record(graft.get("cent")),
            "child_cents": [array_record(value) for value in child_cents],
            "metadata": stable(graft.get("metadata")),
            "sources": stable(graft.get("sources")),
        }
        payload = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        rows.append({
            "index": index,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    projection = {
        "candidate_ids": candidates,
        "candidate_rows": rows,
        "length_debias": bool(getattr(arena, "length_debias", False)),
        "revision_resolution": bool(
            getattr(arena, "revision_resolution", False)),
    }
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema": "grm.det1_4.routing_index_digest.v1",
        "candidate_count": len(candidates),
        "candidate_ids": candidates,
        "projection_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_row_sha256": rows,
        "scope": (
            "active route candidates, centroid/child-centroid values, text/length, "
            "and revision metadata; excludes mounted K/V snapshot fields"
        ),
    }


def verbal_result(answer: str, token_ids: Sequence[int], decode) -> dict[str, Any]:
    trigger = whole_word(answer, "NO")
    latency = None
    if trigger:
        prefix: list[int] = []
        for index, token_id in enumerate(token_ids):
            prefix.append(int(token_id))
            if whole_word(decode(prefix), "NO"):
                latency = int(index)
                break
    return {
        "answer": str(answer),
        "trigger": bool(trigger),
        "latency_token_index": latency,
        "latency_indexing": "zero_based_generated_token",
    }


def _entropy_and_margin(logits: np.ndarray) -> tuple[float, float]:
    row = np.asarray(logits, dtype=np.float64).reshape(-1)
    if row.size < 2 or not np.isfinite(row).all():
        raise DETError("D-ENT received fewer than two finite logits")
    largest = np.partition(row, -2)[-2:]
    largest.sort()
    margin = float(largest[-1] - largest[-2])
    peak = float(row.max())
    exp = np.exp(row - peak)
    denom = float(exp.sum())
    probs = exp / denom
    log_z = peak + math.log(denom)
    entropy = float(log_z - np.dot(probs, row))
    if not (math.isfinite(entropy) and math.isfinite(margin)):
        raise DETError("D-ENT produced a non-finite statistic")
    return entropy, margin


def _effective_order(arena, raw_order: Sequence[int]) -> list[int]:
    return [int(value) for value in arena._resolve_revision_mounts(
        [int(value) for value in raw_order])]


def _revision_component(arena, seed: int, eligible: Sequence[int]) -> set[int]:
    """Return the physical rows belonging to one L2 logical graft.

    L2 resolves a ranked set by explicit ``supersedes`` edges.  Treat those
    edges as undirected only for the purpose of withholding every physical
    revision of the router's one logical rank-1 graft.
    """
    allowed = {int(value) for value in eligible}
    graph = {index: set() for index in allowed}
    for index in allowed:
        metadata = arena.grafts[index].get("metadata") or {}
        linked: set[int] = set()
        for field in ("supersedes", "superseded_by"):
            linked.update(int(value) for value in arena._metadata_node_ids(
                metadata.get(field), len(arena.grafts)))
        for other in linked & allowed:
            graph[index].add(other)
            graph[other].add(index)
    found = {int(seed)}
    pending = [int(seed)]
    while pending:
        index = pending.pop()
        for other in graph.get(index, ()):
            if other not in found:
                found.add(other)
                pending.append(other)
    return found


def route_fixture_profile(
    arena,
    question: str,
    expected: str,
    *,
    live_excluded: set[int],
    route_limit: int,
) -> dict[str, Any]:
    """Freeze raw routing and its production-L2 logical rank-1 graft.

    The literal raw router rank-1 can be an older physical revision.  Required
    production L2 makes its surviving revision head the logical rank-1.  The
    planted unit contains that whole revision component plus exact-value
    mirrors (for example, complete-turn and compact correction rows).  The
    admission filter withholds the complete unit while leaving the full index
    untouched for D-LQR.
    """
    eligible = [
        int(i) for i in arena._route_cand_base()
        if int(i) not in {int(v) for v in live_excluded}
    ]
    if not eligible:
        raise DETError("registered probe has no eligible repository nodes")
    probe_key = arena._probe_key(question)
    raw = list(arena.route(
        question,
        exclude=set(live_excluded),
        limit=len(eligible),
        probe_key=probe_key,
    ) or [])
    effective = _effective_order(arena, raw)
    if not effective:
        raise DETError("registered probe has no post-L2 route candidate")
    rank1_component = _revision_component(arena, int(raw[0]), eligible)
    ranked_component = [int(value) for value in raw if int(value) in rank1_component]
    component_heads = _effective_order(arena, ranked_component)
    if not component_heads:
        raise DETError("raw router rank-1 revision component has no L2 head")
    admission = None
    if bool(getattr(arena, "decisive_admission", False)):
        from core.grm_admission import decisive_admission_profile

        admission = decisive_admission_profile(
            arena,
            question,
            exclude=set(live_excluded),
            route_limit=int(route_limit),
        )
        admission_ranking = [int(value) for value in admission["ranking"]]
        if admission_ranking != [int(value) for value in raw[:len(admission_ranking)]]:
            raise DETError("A-DEC ranking differs from the planted-miss route profile")
    admitted = (
        _effective_order(arena, admission["rank_plan"])
        if admission is not None
        else _effective_order(arena, raw[:int(route_limit)])
    )
    if not admitted:
        raise DETError("registered probe has no production admitted logical graft")
    target = int(component_heads[0])
    target_text = str(arena.grafts[target].get("text", "") or "")
    target_contains_expected = contains_value(target_text, expected)
    aliases = set(rank1_component)
    if target_contains_expected:
        aliases.update(
            int(i) for i in eligible
            if contains_value(str(arena.grafts[int(i)].get("text", "") or ""), expected)
        )
    aliases = sorted(set(aliases))
    if int(raw[0]) not in aliases:
        raise DETError(
            "raw physical rank-1 does not belong to the post-L2 logical "
            f"rank-1 unit: raw={raw[0]} logical={target} aliases={aliases}")
    return {
        "raw_ranking": [int(i) for i in raw],
        "effective_ranking": effective,
        "raw_rank1": int(raw[0]),
        "logical_router_rank1": target,
        "production_admitted_rank1": int(admitted[0]),
        "router_rank1_admitted": bool(set(aliases) & set(admitted)),
        "target_contains_expected": bool(target_contains_expected),
        "logical_alias_ids": aliases,
        "route_backend": str(getattr(arena, "last_route_backend", "unknown")),
        "eligible_count": len(eligible),
        "served_admission_profile": admission,
        "served_effective_admission_plan": admitted,
        "production_route_limit": int(route_limit),
    }


def _lqr_token(arena, q_last: np.ndarray, logical_alias_ids: set[int]) -> dict[str, Any]:
    candidates = [int(i) for i in arena._route_cand_base()]
    base: dict[int, float] = {}
    for index in candidates:
        score = float(arena._cent_score(q_last, arena.grafts[index]))
        if math.isfinite(score):
            base[index] = score
    base = arena._length_debias_scores(base, candidates)
    base = arena._normalize_scores(base) or {}
    raw = sorted(base, key=lambda index: (-float(base[index]), int(index)))
    effective = _effective_order(arena, raw)
    mounted = {int(i) for i in arena.cur_mounts}
    alias_mounted = bool(mounted & logical_alias_ids)

    logical_rows: list[tuple[float, int, bool, list[int]]] = []
    alias_members = [i for i in effective if i in logical_alias_ids]
    if alias_members:
        best_alias = max(alias_members, key=lambda i: (float(base[i]), -int(i)))
        logical_rows.append((
            float(base[best_alias]), int(best_alias), alias_mounted,
            [int(i) for i in alias_members],
        ))
    for index in effective:
        if index in logical_alias_ids:
            continue
        logical_rows.append((float(base[index]), int(index), index in mounted, [int(index)]))
    logical_rows.sort(key=lambda row: (-row[0], row[1]))
    mounted_rows = [row for row in logical_rows if row[2]]
    unmounted_rows = [row for row in logical_rows if not row[2]]
    best_mounted = mounted_rows[0] if mounted_rows else None
    best_unmounted = unmounted_rows[0] if unmounted_rows else None
    trigger = bool(
        best_unmounted is not None
        and (best_mounted is None or best_unmounted[0] > best_mounted[0])
    )
    tied = bool(
        best_unmounted is not None and best_mounted is not None
        and best_unmounted[0] == best_mounted[0]
    )
    return {
        "trigger": trigger,
        "strict_outrank": True,
        "tie": tied,
        "best_unmounted_id": None if best_unmounted is None else best_unmounted[1],
        "best_unmounted_score": None if best_unmounted is None else best_unmounted[0],
        "best_mounted_id": None if best_mounted is None else best_mounted[1],
        "best_mounted_score": None if best_mounted is None else best_mounted[0],
        "effective_index_size": len(logical_rows),
    }


@dataclass
class DetectorObserver:
    """Read-only GPT-OSS observer on one generation attempt."""

    arena: Any
    logical_alias_ids: set[int]
    ngen: int
    active_detectors: frozenset[str] = frozenset(MECHANISTIC)
    ngh_schedule: str = "legacy_inline"

    def __post_init__(self) -> None:
        self.logical_alias_ids = {int(i) for i in self.logical_alias_ids}
        self.active_detectors = frozenset(str(value) for value in self.active_detectors)
        unknown = self.active_detectors - frozenset(MECHANISTIC)
        if unknown:
            raise DETError(f"unknown mechanistic detector hook(s): {sorted(unknown)}")
        if self.ngh_schedule != "legacy_inline":
            raise DETError(f"unsupported D-NGH schedule: {self.ngh_schedule}")
        self.records: list[dict[str, Any]] = []
        self._original_forward = None
        self._gpt = None
        self._original_sink = None
        self._original_sliding = None
        self._route_attn = None
        self._route_capture_present = False
        self._route_capture_value = None
        self._route_captured_present = False
        self._route_captured_value = None
        self._inside_forward = False
        self._suppress_sink = False
        self._mass_rows: list[dict[str, float]] = []
        self.expected_full_layers = 0

    def __enter__(self):
        from core import gpt_oss20b_tc as gpt

        if "D-NGH" in self.active_detectors:
            full = [
                layer for layer in self.arena.m.layers
                if layer.self_attn.layer_type == "full_attention"
            ]
            bad = [
                int(layer.self_attn.layer_idx) for layer in full
                if str(layer.self_attn.attention_mode) != "standard"
            ]
            if bad:
                raise DETError(
                    "D-NGH observer supports the current standard GPT-OSS path only; "
                    f"non-standard full layers: {bad}")
            self.expected_full_layers = len(full)
            if self.expected_full_layers <= 0:
                raise DETError("D-NGH found no GPT-OSS full-attention layers")

        self._gpt = gpt
        self._original_sink = gpt.sink_attention_tc
        self._original_sliding = gpt.sliding_sink_attention_tc
        self._original_forward = self.arena._forward
        if "D-LQR" in self.active_detectors:
            self._route_attn = self.arena.m.layers[int(self.arena.route_layer)].self_attn
            self._route_capture_present = hasattr(self._route_attn, "_capture_q")
            self._route_capture_value = getattr(self._route_attn, "_capture_q", None)
            self._route_captured_present = hasattr(self._route_attn, "_captured_q")
            self._route_captured_value = getattr(self._route_attn, "_captured_q", None)
            self._route_attn._capture_q = True

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
            # The preceding production route capture clears this opt-in flag.
            # Re-arm at each answer-position forward so D-LQR sees the live
            # contextual stream and not the route probe (or ``None``).
            if "D-LQR" in observer.active_detectors:
                observer._route_attn._capture_q = True
                observer._route_attn._captured_q = None
            observer._inside_forward = "D-NGH" in observer.active_detectors
            try:
                logits = observer._original_forward(ids, last_only=last_only)
            finally:
                observer._inside_forward = False
            lqr = None
            if "D-LQR" in observer.active_detectors:
                q = np.asarray(observer._route_attn._captured_q, dtype=np.float32)
                if q.ndim != 4 or q.shape[0] != 1:
                    raise DETError(f"unexpected live query capture shape: {q.shape}")
                q_last = q[0, :, -1:, :]
                lqr = _lqr_token(observer.arena, q_last, observer.logical_alias_ids)
            ent = None
            prediction_token_id = int(np.asarray(logits).argmax())
            if "D-ENT" in observer.active_detectors:
                entropy, margin = _entropy_and_margin(logits)
                ent = {"entropy_nats": entropy, "top1_top2_margin": margin}

            mass = None
            if "D-NGH" in observer.active_detectors:
                mass = observer._summarize_mass(observer._mass_rows)
            observer.records.append({
                "token_index": len(observer.records),
                "prediction_token_id": prediction_token_id,
                "lqr": lqr,
                "ngh": mass,
                "ent": ent,
                "full_attention_layers": (
                    observer.expected_full_layers if mass is not None else 0),
            })
            return logits

        if "D-NGH" in self.active_detectors:
            gpt.sink_attention_tc = sink_wrapper
            gpt.sliding_sink_attention_tc = sliding_wrapper
        self.arena._forward = forward_wrapper
        return self

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

    def _validate_mass_count(self, values: Sequence[Any]) -> None:
        if len(values) != self.expected_full_layers:
            raise DETError(
                "D-NGH full-layer capture count mismatch: "
                f"expected {self.expected_full_layers}, observed {len(values)}")

    def _summarize_mass(
        self, rows: Sequence[Mapping[str, float]],
    ) -> dict[str, float]:
        self._validate_mass_count(rows)
        mass = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "physical_sink_mass", "mounted_mass", "live_mass",
                "learned_sink_mass",
            )
        }
        total = sum(mass.values())
        if not (0.98 <= total <= 1.02):
            raise DETError(f"D-NGH mass partition does not sum to one: {mass}")
        return mass

    def __exit__(self, exc_type, exc, tb):
        if self._original_forward is not None:
            self.arena._forward = self._original_forward
        if self._gpt is not None:
            self._gpt.sink_attention_tc = self._original_sink
            self._gpt.sliding_sink_attention_tc = self._original_sliding
        if self._route_attn is not None:
            if self._route_capture_present:
                self._route_attn._capture_q = self._route_capture_value
            elif hasattr(self._route_attn, "_capture_q"):
                delattr(self._route_attn, "_capture_q")
            if self._route_captured_present:
                self._route_attn._captured_q = self._route_captured_value
            elif hasattr(self._route_attn, "_captured_q"):
                delattr(self._route_attn, "_captured_q")
        return False

    def finish(self) -> dict[str, Any]:
        # If generation reaches ngen without a stop, ArenaCache._attempt runs
        # one extra forward solely to commit the final predicted token to KV.
        # Its logits are unused and are not an answer position.
        kept = list(self.records)
        dropped_flush = False
        if len(kept) == int(self.ngen) + 1:
            kept = kept[:-1]
            dropped_flush = True
        if len(kept) > int(self.ngen):
            raise DETError(
                f"observer captured {len(kept)} answer rows for ngen={self.ngen}")
        for index, row in enumerate(kept):
            row["token_index"] = int(index)
        return {
            "schema": "grm.det1.signals.v1",
            "active_detectors": sorted(self.active_detectors),
            "ngh_schedule": (
                self.ngh_schedule if "D-NGH" in self.active_detectors else None),
            "token_indexing": "zero_based_generated_prediction",
            "tokens": kept,
            "prediction_token_ids": [
                int(row["prediction_token_id"]) for row in kept
            ],
            "unused_final_cache_flush_dropped": dropped_flush,
            "lqr_semantics": "contextual_pre_rope_q_semantic_only_post_l2_logical_index",
            "ngh_semantics": "actual_standard_attention_operands_full_layers_only_absolute_mounted_probability",
            "ent_semantics": "exact_host_fp32_logit_row_entropy_and_top1_top2_margin",
        }


def fit_thresholds(calibration_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    served = [row for row in calibration_rows if row.get("variant") == "served"]
    if not served:
        raise DETError("threshold fit has no served calibration rows")
    ngh_turn_minima = []
    ent_turn_maxima = []
    margin_turn_minima = []
    for row in served:
        tokens = list((row.get("signals") or {}).get("tokens") or [])
        if not tokens:
            raise DETError(f"calibration row has no answer tokens: {row.get('row_id')}")
        ngh_turn_minima.append(min(float(t["ngh"]["mounted_mass"]) for t in tokens))
        ent_turn_maxima.append(max(float(t["ent"]["entropy_nats"]) for t in tokens))
        margin_turn_minima.append(min(float(t["ent"]["top1_top2_margin"]) for t in tokens))
    return {
        "schema": "grm.det1.thresholds.v1",
        "created_utc": utc_now(),
        "created_unix_ns": __import__("time").time_ns(),
        "fit_fixture_ids": [str(row["fixture_id"]) for row in served],
        "fit_row_ids": [str(row["row_id"]) for row in served],
        "fit_side": "served_controls_only",
        "D-NGH": {
            "score": "per-token full-layer mean mounted_mass",
            "direction": "trigger_if_strictly_below",
            "threshold": float(min(ngh_turn_minima)),
            "selection": "minimum_of_served_turn_minima_strict_envelope",
            "fit_turn_minima": ngh_turn_minima,
        },
        "D-ENT": {
            "composite": "entropy_high_OR_margin_low",
            "entropy_direction": "trigger_if_strictly_above",
            "entropy_threshold": float(max(ent_turn_maxima)),
            "margin_direction": "trigger_if_strictly_below",
            "margin_threshold": float(min(margin_turn_minima)),
            "selection": "served_turn_extrema_strict_envelope",
            "fit_turn_entropy_maxima": ent_turn_maxima,
            "fit_turn_margin_minima": margin_turn_minima,
        },
    }


def detector_decision(
    row: Mapping[str, Any],
    detector: str,
    thresholds: Mapping[str, Any],
) -> tuple[bool, int | None]:
    if detector == "D-VERB":
        value = row.get("verbal") or {}
        return bool(value.get("trigger")), value.get("latency_token_index")
    tokens = list((row.get("signals") or {}).get("tokens") or [])
    for token in tokens:
        index = int(token["token_index"])
        if detector == "D-LQR" and bool(token["lqr"]["trigger"]):
            return True, index
        if detector == "D-NGH":
            threshold = float(thresholds["D-NGH"]["threshold"])
            if float(token["ngh"]["mounted_mass"]) < threshold:
                return True, index
        if detector == "D-ENT":
            ent = thresholds["D-ENT"]
            if (
                float(token["ent"]["entropy_nats"]) > float(ent["entropy_threshold"])
                or float(token["ent"]["top1_top2_margin"]) < float(ent["margin_threshold"])
            ):
                return True, index
    return False, None


def race_metrics(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], str]:
    positives = [row for row in rows if row.get("variant") == "planted_miss"]
    negatives = [row for row in rows if row.get("variant") == "served"]
    # GRM-DET1.11: the race runs at the ACHIEVED lawful pair count, with a
    # registered floor of 8 pairs per arm below which the campaign refuses
    # rather than run a hollow race.  The arms must also be balanced — an
    # unequal race has no honest recall/FPR pairing.
    if len(positives) < ACHIEVED_EVAL_PAIR_FLOOR or (
        len(negatives) < ACHIEVED_EVAL_PAIR_FLOOR
    ):
        raise DETError(
            f"DET1.11 achieved-pair floor not met: race requires at least "
            f"{ACHIEVED_EVAL_PAIR_FLOOR}+{ACHIEVED_EVAL_PAIR_FLOOR} rows, "
            f"got {len(positives)}+{len(negatives)}")
    if len(positives) != len(negatives):
        raise DETError(
            f"race arms are unbalanced: {len(positives)} planted-miss vs "
            f"{len(negatives)} served")
    table = []
    for detector in DETECTORS:
        pos = [detector_decision(row, detector, thresholds) for row in positives]
        neg = [detector_decision(row, detector, thresholds) for row in negatives]
        tp = sum(int(trigger) for trigger, _latency in pos)
        fp = sum(int(trigger) for trigger, _latency in neg)
        recall = tp / len(positives)
        fpr = fp / len(negatives)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        latencies = [int(latency) for trigger, latency in pos
                     if trigger and latency is not None]
        median_latency = float(np.median(latencies)) if latencies else None
        table.append({
            "detector": detector,
            "tp": tp,
            "positive_n": len(positives),
            "fp": fp,
            "negative_n": len(negatives),
            "recall": recall,
            "false_positive_rate": fpr,
            "precision": precision,
            "f1": f1,
            "median_trigger_latency_token": median_latency,
            "latency_population": "triggered_planted_misses",
            "viable": bool(recall >= 0.80 and fpr <= 0.10),
        })
    viable = [row for row in table if row["viable"]]
    winners: list[str] = []
    if viable:
        best = max(float(row["f1"]) for row in viable)
        winners = [str(row["detector"]) for row in viable if float(row["f1"]) == best]
    by_name = {str(row["detector"]): row for row in table}
    best_mechanistic = max(float(by_name[name]["f1"]) for name in MECHANISTIC)
    verbal = float(by_name["D-VERB"]["f1"])
    viable_mech = [name for name in MECHANISTIC if by_name[name]["viable"]]
    verbal_winner = "D-VERB" in winners
    verbal_ties_best_mechanistic = verbal == best_mechanistic
    if verbal_winner or verbal_ties_best_mechanistic:
        verdict = "REFUTED"
    elif "D-LQR" in winners and all(
        float(by_name[name]["f1"]) > verbal for name in viable_mech
    ):
        verdict = "SUPPORTED"
    elif best_mechanistic > verbal and "D-LQR" not in winners:
        verdict = "PARTIAL"
    else:
        # The order's three iff clauses do not cover this corner (for
        # example, D-LQR wins but another viable mechanism trails D-VERB).
        # Preserve that gap instead of relabeling it after seeing data.
        verdict = "UNADJUDICATED"
    return table, winners, verdict


def selftest() -> dict[str, Any]:
    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    entropy, margin = _entropy_and_margin(logits)
    if not (0.0 < entropy < math.log(3.0) and margin == 1.0):
        raise DETError("entropy/margin selftest failed")
    verbal = verbal_result("NO.", [1, 2], lambda ids: "N" if len(ids) == 1 else "NO")
    if verbal != {
        "answer": "NO.", "trigger": True, "latency_token_index": 1,
        "latency_indexing": "zero_based_generated_token",
    }:
        raise DETError(f"verbal parser selftest failed: {verbal}")
    fake_cal = [{
        "row_id": "cal:served", "fixture_id": "cal", "variant": "served",
        "signals": {"tokens": [
            {"ngh": {"mounted_mass": 0.4},
             "ent": {"entropy_nats": 1.0, "top1_top2_margin": 2.0}},
            {"ngh": {"mounted_mass": 0.3},
             "ent": {"entropy_nats": 1.5, "top1_top2_margin": 1.0}},
        ]},
    }]
    thresholds = fit_thresholds(fake_cal)
    if thresholds["D-NGH"]["threshold"] != 0.3:
        raise DETError("threshold selftest failed")

    class _LineageArena:
        decisive_admission = False
        last_route_backend = "python"

        def __init__(self):
            self.grafts = [
                {"text": "old value", "metadata": {
                    "supersedes": [], "superseded_by": [2]}},
                {"text": "unrelated", "metadata": {
                    "supersedes": [], "superseded_by": []}},
                {"text": "current code is Needle-7", "metadata": {
                    "supersedes": [0], "superseded_by": []}},
            ]

        def _route_cand_base(self):
            return [0, 1, 2]

        def _probe_key(self, _question):
            return np.zeros((1,), dtype=np.float32)

        def route(self, _question, *, exclude, limit, probe_key):
            del exclude, probe_key
            return [0, 1, 2][:limit]

        def _metadata_node_ids(self, values, _count):
            return [int(value) for value in (values or [])]

        def _resolve_revision_mounts(self, picks):
            picks = [int(value) for value in picks]
            return [value for value in picks if not (value == 0 and 2 in picks)]

    lineage = route_fixture_profile(
        _LineageArena(), "question", "Needle-7",
        live_excluded=set(), route_limit=3)
    if not (
        lineage["raw_rank1"] == 0
        and lineage["logical_router_rank1"] == 2
        and lineage["logical_alias_ids"] == [0, 2]
        and lineage["router_rank1_admitted"] is True
    ):
        raise DETError(f"logical rank-1 lineage selftest failed: {lineage}")
    return {
        "schema": "grm.det1.cpu_selftest.v1",
        "status": "PASS",
        "checks": {
            "finite_canonical_writer": "PASS",
            "entropy_margin": "PASS",
            "verbal_exact_NO_parser": "PASS",
            "strict_fit_side_thresholds": "PASS",
            "raw_rank1_l2_lineage": "PASS",
        },
    }


if __name__ == "__main__":
    print(json.dumps(selftest(), sort_keys=True))
