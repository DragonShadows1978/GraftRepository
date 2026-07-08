#!/usr/bin/env python3
"""Composed GPT-OSS + GRM end-to-end session receipt driver.

P1 Leg 2 driver for docs/GRM_E2E_RECEIPT_PLAN.md. The full run is left for
the lead; use ``--mode smoke`` for the bounded 8-10 turn proof.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core.gpt_oss20b_tc import (  # noqa: E402
    GptOss20B_TC,
    gpt_oss_grm_dialect_kwargs,
)
from core.graft_arena import GQAArenaCache  # noqa: E402
from core.graft_repository import GraftRepository  # noqa: E402


SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
NATIVE_LIB = ROOT / "cpp" / "build" / "libgrm_runtime.so"
# Harmony SYS sink (Leg-1 / diagnosis-validated). n_sink derives from this;
# default ArenaCache sink "<conversation>\n" mis-seats GPT-OSS YARN.
HARMONY_SINK = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|>"
)
SYSTEM_PREFIX = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|><|start|>user<|message|>"
)
ASSISTANT_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"
HARMONY_STOPS = (
    "<|return|>", "<|end|>", "<|start|>user", "<|start|>assistant")


class GptOssGQAArenaCache(GQAArenaCache):
    POSITION_LAW = "rope_full_yarn"
    STATE_KIND = "kv"
    GRAFTABILITY = "seat_remountable"
    REMOUNTABLE = True
    COMPOSITION = "multi_mount"


def harmony_turn(user_text: str, assistant_text: str | None) -> str:
    prompt = f"{SYSTEM_PREFIX}{user_text}{ASSISTANT_FINAL}"
    if assistant_text is None:
        return prompt
    return f"{prompt}{assistant_text}<|end|>"


def plant_acceptance(event: dict[str, Any]) -> str:
    """Scripted assistant acceptance for Leg-1 feed() plant turns.

    Value-bearing complete text — never free-generated. Embeds the fact
    phrase so the deposited K/V carries the payload (not a refusal).
    """
    phrase = event.get("fact_phrase")
    if not phrase:
        phrase = f"current {event['fact_id']} value is {event['value']}"
    return f"Understood — the {phrase}."


def fact_turn(fact_id: str, value: str, *, label: str | None = None) -> dict[str, Any]:
    label = label or fact_id
    phrase = f"current {fact_id} value is {value}"
    event = {
        "kind": "fact",
        "fact_id": fact_id,
        "value": value,
        "user": (
            f"Memory planting turn. fact {fact_id}. The {phrase}. "
            f"If asked later for {label}, answer {value} only."
        ),
        "fact_phrase": phrase,
    }
    event["assistant"] = plant_acceptance(event)
    return event


def supersede_turn(fact_id: str, old: str, new: str) -> dict[str, Any]:
    old_phrase = f"current {fact_id} value is {old}"
    new_phrase = f"current {fact_id} value is {new}"
    event = {
        "kind": "supersede",
        "fact_id": fact_id,
        "old_value": old,
        "value": new,
        "user": (
            f"Authoritative update turn. fact {fact_id}. The {new_phrase}. "
            f"This is the current value. If asked later, answer {new} only."
        ),
        "correction_command": f"correct memory: {old_phrase} => The {new_phrase}.",
        "fact_phrase": new_phrase,
    }
    event["assistant"] = plant_acceptance(event)
    return event


def filler_turn(tag: str) -> dict[str, Any]:
    return {
        "kind": "filler",
        "user": (
            f"Session continuity filler {tag}. Acknowledge briefly. "
            "Do not change any stored fact values."
        ),
    }


def probe_turn(
    fact_id: str,
    expected: str,
    *,
    source_turn: int,
    supersession: bool = False,
    old_value: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "probe",
        "fact_id": fact_id,
        "expected": expected,
        "accepts": [expected],
        "old_value": old_value,
        "source_turn": int(source_turn),
        "supersession": bool(supersession),
        "user": (
            f"Recall probe. What is the current {fact_id} value? "
            "Reply with only the value."
        ),
    }


def build_smoke_script() -> list[dict[str, Any]]:
    # Fork-A code-shaped values (exact map from
    # artifacts/grm_e2e/smoke_session_20260708_154313/run_config.json).
    return [
        fact_turn("orion pin", "Auric-4-Alpha"),
        filler_turn("smoke alpha"),
        supersede_turn("orion pin", "Auric-4-Alpha", "Kestrel-9-Tango"),
        fact_turn("cypher bridge", "Vortex-3-Sierra"),
        filler_turn("smoke beta"),
        probe_turn("cypher bridge", "Vortex-3-Sierra", source_turn=3),
        probe_turn(
            "orion pin", "Kestrel-9-Tango", source_turn=2,
            supersession=True, old_value="Auric-4-Alpha"),
        filler_turn("smoke gamma"),
        filler_turn("smoke delta"),
        filler_turn("smoke epsilon"),
    ]


def build_full_script() -> list[dict[str, Any]]:
    # Smoke values recovered from run_config; remaining facts re-derived
    # in the same Word-N-Phonetic code-shaped style.
    return [
        fact_turn("orion pin", "Auric-4-Alpha"),
        filler_turn("full alpha"),
        fact_turn("lyra dock", "Nadir-1-Delta"),
        filler_turn("full beta"),
        fact_turn("cypher bridge", "Vortex-3-Sierra"),
        probe_turn("orion pin", "Auric-4-Alpha", source_turn=0),
        supersede_turn("orion pin", "Auric-4-Alpha", "Kestrel-9-Tango"),
        fact_turn("nova key", "Quartz-5-Bravo"),
        filler_turn("full gamma"),
        probe_turn("cypher bridge", "Vortex-3-Sierra", source_turn=4),
        fact_turn("mira seal", "Silver-6-Charlie"),
        supersede_turn("lyra dock", "Nadir-1-Delta", "Zenith-2-Echo"),
        filler_turn("full delta"),
        probe_turn(
            "orion pin", "Kestrel-9-Tango", source_turn=6,
            supersession=True, old_value="Auric-4-Alpha"),
        fact_turn("terra port", "Harbor-8-Golf"),
        filler_turn("full epsilon"),
        probe_turn(
            "lyra dock", "Zenith-2-Echo", source_turn=11,
            supersession=True, old_value="Nadir-1-Delta"),
        supersede_turn("mira seal", "Silver-6-Charlie", "Gold-7-Foxtrot"),
        filler_turn("full zeta"),
        probe_turn("nova key", "Quartz-5-Bravo", source_turn=7),
        fact_turn("ember code", "Violet-2-Hotel"),
        filler_turn("full eta"),
        probe_turn(
            "mira seal", "Gold-7-Foxtrot", source_turn=17,
            supersession=True, old_value="Silver-6-Charlie"),
        filler_turn("full theta"),
        probe_turn("terra port", "Harbor-8-Golf", source_turn=14),
        filler_turn("full iota"),
        probe_turn("ember code", "Violet-2-Hotel", source_turn=20),
        fact_turn("atlas tone", "Cobalt-1-India"),
        filler_turn("full kappa"),
        filler_turn("full lambda"),
        probe_turn("atlas tone", "Cobalt-1-India", source_turn=27),
        filler_turn("full mu"),
        fact_turn("polaris mark", "Marble-4-Juliet"),
        filler_turn("full nu"),
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--session-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=Path(SNAPSHOT))
    p.add_argument("--native-lib", type=Path, default=NATIVE_LIB)
    p.add_argument("--ngen", type=int, default=None)
    p.add_argument("--max-trips", type=int, default=1)
    p.add_argument("--live-turns", type=int, default=None)
    # Leg-1 proven / diagnosis: width 384 → live_shift≈387 collapses GPT-OSS.
    p.add_argument("--arena-width", type=int, default=96)
    p.add_argument("--max-live", type=int, default=4096)
    # Fork A: production-realistic multi-mount (top-k 2-3, not argmax-only).
    # Arena.step already slices ranking into self.topk mounts; this flag is
    # the driver call-site width of that slice (default 3 for this receipt).
    p.add_argument("--topk", type=int, default=3,
                   help="route multi-mount count (arena.topk); Fork-A default 3")
    p.add_argument("--restart-after", type=int, default=None)
    p.add_argument("--skip-gpu-idle-check", action="store_true")
    return p.parse_args(argv)


def default_session_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "grm_e2e" / f"{mode}_session_{stamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def active_compute_pids() -> list[str]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def wait_for_idle_gpu(max_wait_s: int = 900) -> dict[str, Any]:
    waited = 0
    checks = []
    while True:
        pids = active_compute_pids()
        checks.append({"waited_s": waited, "pids": pids})
        if not pids:
            return {"idle": True, "waited_s": waited, "checks": checks}
        if waited >= max_wait_s:
            return {"idle": False, "waited_s": waited, "checks": checks}
        print(f"GPU busy with compute PIDs {pids}; waiting 60s", flush=True)
        time.sleep(60)
        waited += 60


def vram_snapshot() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    raw = proc.stdout.strip()
    out: dict[str, Any] = {"raw": raw, "returncode": int(proc.returncode)}
    if proc.returncode == 0 and raw:
        rows = []
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                rows.append({"used_mb": int(parts[0]), "total_mb": int(parts[1])})
        out["gpus"] = rows
    else:
        out["stderr"] = proc.stderr.strip()
    return out


class TurnTimers:
    def __init__(self) -> None:
        self.route_ms = 0.0
        self.route_calls = 0
        self.deposit_ms = 0.0
        self.deposit_calls = 0
        self.mount_ms = 0.0
        self.mount_calls = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_wall_ms": self.route_ms,
            "route_calls": self.route_calls,
            "deposit_wall_ms": self.deposit_ms,
            "deposit_calls": self.deposit_calls,
            "mount_wall_ms": self.mount_ms,
            "mount_calls": self.mount_calls,
        }


def install_turn_timers(arena) -> tuple[TurnTimers, Any]:
    timers = TurnTimers()
    original = {
        "route": arena.route,
        "swap": arena.swap,
        "deposit": arena.deposit,
        "deposit_from_cache": arena.deposit_from_cache,
    }

    def route_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["route"](*args, **kwargs)
        finally:
            timers.route_ms += (time.perf_counter() - t0) * 1000.0
            timers.route_calls += 1

    def swap_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["swap"](*args, **kwargs)
        finally:
            timers.mount_ms += (time.perf_counter() - t0) * 1000.0
            timers.mount_calls += 1

    def deposit_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["deposit"](*args, **kwargs)
        finally:
            timers.deposit_ms += (time.perf_counter() - t0) * 1000.0
            timers.deposit_calls += 1

    def deposit_from_cache_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["deposit_from_cache"](*args, **kwargs)
        finally:
            timers.deposit_ms += (time.perf_counter() - t0) * 1000.0
            timers.deposit_calls += 1

    arena.route = route_wrapper
    arena.swap = swap_wrapper
    arena.deposit = deposit_wrapper
    arena.deposit_from_cache = deposit_from_cache_wrapper

    def restore() -> None:
        arena.route = original["route"]
        arena.swap = original["swap"]
        arena.deposit = original["deposit"]
        arena.deposit_from_cache = original["deposit_from_cache"]

    return timers, restore


def contains_accept(answer: str, values: list[str]) -> bool:
    for value in values:
        pattern = r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])"
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def score_probe(answer: str, event: dict[str, Any]) -> dict[str, Any]:
    accepts = [str(x) for x in event.get("accepts", ())]
    old_value = event.get("old_value")
    hit = contains_accept(answer, accepts)
    stale_hit = contains_accept(answer, [str(old_value)]) if old_value else False
    return {
        "fact_id": event["fact_id"],
        "expected": event["expected"],
        "accepts": accepts,
        "old_value": old_value,
        "answer": answer,
        "pass": bool(hit and not stale_hit),
        "contains_expected": bool(hit),
        "contains_stale": bool(stale_hit),
    }


def collect_route_diagnostics(
    arena,
    probe_text: str,
    source_node_id: int | None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Driver-side top-K route ranking for probe turns (ids + scores).

    Uses the same arena.route() entry as step(), then recomputes Python-side
    scores for the returned nodes (native/CUDA path returns ids only).
    """
    import math

    live_idx = {g for g, _ in arena.live_segs if g is not None}
    n = len(getattr(arena, "grafts", ()) or ())
    limit = max(int(top_k), n) if n else int(top_k)
    ranking = list(arena.route(probe_text, exclude=live_idx, limit=limit) or [])
    backend = getattr(arena, "last_route_backend", None)

    scores: dict[int, float] = {}
    try:
        p = arena._probe_key(probe_text)
        qrare = arena._rare_tokens(probe_text)
        cand = list(ranking) if ranking else [
            i for i in range(n) if i not in live_idx]
        base = arena._vector_route_scores(p, cand)
        if base is None:
            base = {}
            for i in cand:
                sc = arena._cent_score(p, arena.grafts[i])
                if sc == sc:  # finite
                    base[i] = float(sc)
        base = arena._normalize_scores(base) or {}
        for i in cand:
            if i not in base:
                continue
            sc = float(base[i]) + float(arena._lex_bonus(qrare, arena.grafts[i]))
            if math.isfinite(sc):
                scores[int(i)] = sc
    except Exception as err:  # diagnostics must not abort the turn
        scores = {}
        score_err = repr(err)
    else:
        score_err = None

    top = []
    for rank, idx in enumerate(ranking[: int(top_k)], start=1):
        idx_i = int(idx)
        g = arena.grafts[idx_i] if 0 <= idx_i < n else {}
        text = str(g.get("text", "") or "")
        top.append({
            "rank": rank,
            "node_id": idx_i,
            "score": scores.get(idx_i),
            "text_prefix": text[:120],
            "contains_value_hint": bool(
                re.search(r"[A-Za-z]+-\d+-[A-Za-z]+", text)),
        })

    source_rank = None
    if source_node_id is not None:
        try:
            source_rank = ranking.index(int(source_node_id)) + 1
        except ValueError:
            source_rank = None

    return {
        "top5": top,
        "source_node_id": (
            int(source_node_id) if source_node_id is not None else None),
        "source_rank": source_rank,
        "ranking_len": len(ranking),
        "route_backend": backend,
        "score_error": score_err,
        "live_excluded": sorted(int(x) for x in live_idx),
        # Full rank order kept so max-pool length bias stays visible even
        # when the mount slice is wider than 1 (Fork A: do not hide bias).
        "ranking_ids": [int(x) for x in ranking],
    }


def _budget_fit_mounts(arena, picks: list[int]) -> list[int]:
    """Pack mounts into arena.width in the given order (rank order)."""
    budget = int(arena.width)
    out: list[int] = []
    used = 0
    for i in picks:
        n = int(arena.grafts[int(i)]["ntok"])
        if used + n <= budget:
            out.append(int(i))
            used += n
    return out


def probe_multimount_chat(
    repo: GraftRepository,
    user_text: str,
    *,
    topk: int,
    ngen: int,
) -> tuple[str, dict[str, Any]]:
    """Probe path: mount route top-k via arena multi-mount, not argmax-only.

    Arena.step's precise-first policy collapses identifier probes to rank-1
    alone when rank-1 covers all probe rare tokens. Fork A wants the
    production multi-mount slice (top-k) seated so a correct memory at
    ranks 2-3 is present even when rank-1 is a length-biased distractor.
    Diagnostics still record full ranking / source_rank separately.
    """
    arena = repo.arena
    before = repo._snapshot_state()
    live_idx = {g for g, _ in arena.live_segs if g is not None}
    want = max(int(topk), 1)
    ranking = list(arena.route(user_text, exclude=live_idx, limit=want) or [])
    planned = [int(x) for x in ranking[:want]]
    fitted = _budget_fit_mounts(arena, planned)
    # Seat in sorted order (matches arena._attempt / swap convention).
    picks = sorted(fitted)
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    stops = arena.stop_sequences or ()
    ans, info = arena._attempt(user_text, picks, int(ngen), True, stops)
    info = dict(info or {})
    info["trip"] = 0
    info["driver_probe_multimount"] = True
    info["driver_topk"] = int(want)
    info["mount_plan"] = planned
    info["mount_fitted"] = picks
    info["mount_dropped_for_width"] = [
        i for i in planned if i not in set(picks)]
    extracted = repo._extract_from_new_turns(
        before,
        context={
            "event": "chat",
            "user_text": user_text,
            "assistant_text": ans,
        },
    )
    if extracted:
        info["extraction"] = extracted
    repo.runtime._finish_turn_event(
        "chat", before, extraction=extracted, autosave=True)
    return ans, info


def script_for_mode(mode: str) -> list[dict[str, Any]]:
    return build_smoke_script() if mode == "smoke" else build_full_script()


def repo_node_count(repo: GraftRepository) -> int:
    return len(repo.arena.grafts)


def live_node_ids(repo: GraftRepository) -> list[int]:
    return [int(g) for g, _ntok in repo.arena.live_segs if g is not None]


def load_model_and_repo(args: argparse.Namespace, session_dir: Path):
    model, model_info = GptOss20B_TC.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    cfg = model.config
    dialect = gpt_oss_grm_dialect_kwargs(cfg)
    arena_kw = {
        "route_layer": int(dialect["route_layer"]),
        "arena_width": int(args.arena_width),
        "topk": int(args.topk),
        "live_turns": int(args.live_turns),
        "max_live": int(args.max_live),
        "sink_text": HARMONY_SINK,
        "prompt_template": harmony_turn,
        "stop_sequences": HARMONY_STOPS,
        "storage_bits": 8,
    }
    repo = GraftRepository(
        model,
        encode,
        decode,
        str(session_dir / "repository"),
        autosave=False,
        arena_cls=GptOssGQAArenaCache,
        native_lib_path=str(args.native_lib) if args.native_lib else None,
        native_auto=False,
        **arena_kw,
    )
    return model, tokenizer, repo, model_info


def refeed_live_window(repo: GraftRepository, transcript: list[dict[str, Any]],
                       live_turns: int) -> list[dict[str, Any]]:
    replayed = []
    for row in transcript[-int(live_turns):]:
        turn_text = harmony_turn(row["user"], row["assistant"])
        before = len(repo.arena.live_segs)
        repo.arena.feed(turn_text, deposit=False)
        if len(repo.arena.live_segs) > before and row.get("chat_node_id") is not None:
            _gidx, ntok = repo.arena.live_segs[-1]
            repo.arena.live_segs[-1] = (int(row["chat_node_id"]), ntok)
        replayed.append({
            "turn": int(row["turn"]),
            "chat_node_id": row.get("chat_node_id"),
            "ntok": int(repo.arena.live_segs[-1][1]) if repo.arena.live_segs else 0,
        })
    return replayed


def stage_paths(session_dir: Path) -> dict[str, Path]:
    return {
        "config": session_dir / "run_config.json",
        "transcript": session_dir / "transcript.jsonl",
        "instrumentation": session_dir / "instrumentation.jsonl",
        "scorecard": session_dir / "probe_scorecard.json",
        "summary": session_dir / "summary.json",
        "restart": session_dir / "restart.json",
    }


def plant_complete_turn(
    repo: GraftRepository,
    event: dict[str, Any],
) -> tuple[str, dict[str, Any], int | None]:
    """Leg-1 feed() COMPLETE plant: user + scripted acceptance, deposit/evict.

    No free generation. Deposit path is the same arena.feed(deposit=True)
    used by runtime.add_turn, but with the Harmony complete-turn text so the
    harvested K/V is value-bearing.
    """
    answer = event.get("assistant") or plant_acceptance(event)
    turn_text = harmony_turn(event["user"], answer)
    before = repo._snapshot_state()
    nodes_before = len(repo.arena.grafts)
    repo.arena.feed(turn_text, deposit=True)
    repo._set_new_node_provenance(before, "exchange_span")
    extracted = repo._extract_from_new_turns(
        before,
        context={
            "event": "plant_feed",
            "user_text": event["user"],
            "assistant_text": answer,
        },
    )
    result = repo.runtime._finish_turn_event(
        "add_turn", before, extraction=extracted, autosave=False)
    new_nodes = list(getattr(result, "new_nodes", ()) or ())
    chat_node_id = int(new_nodes[0]) if new_nodes else None
    # Prefer the live-seg graft if feed assigned one.
    if repo.arena.live_segs:
        gidx, _ntok = repo.arena.live_segs[-1]
        if gidx is not None:
            chat_node_id = int(gidx)
    deposited_text = ""
    if chat_node_id is not None and 0 <= chat_node_id < len(repo.arena.grafts):
        deposited_text = str(repo.arena.grafts[chat_node_id].get("text", "") or "")
    value = str(event.get("value") or "")
    phrase = str(event.get("fact_phrase") or "")
    info = {
        "plant_mode": "feed_complete",
        "mounts": list(repo.arena.cur_mounts),
        "live_tokens": int(sum(n for _g, n in repo.arena.live_segs)),
        "new_nodes": [int(x) for x in new_nodes],
        "nodes_before": int(nodes_before),
        "nodes_after": int(len(repo.arena.grafts)),
        "deposit_contains_value": bool(value and value in deposited_text),
        "deposit_contains_phrase": bool(phrase and phrase in deposited_text),
        "deposit_text_prefix": deposited_text[:200],
        "extraction": list(extracted or ()),
    }
    return answer, info, chat_node_id


def run_turn(
    repo: GraftRepository,
    event: dict[str, Any],
    turn_idx: int,
    *,
    paths: dict[str, Path],
    transcript: list[dict[str, Any]],
    turn_records: dict[int, dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    resumed: bool,
) -> None:
    mounts_before = list(repo.arena.cur_mounts)
    nodes_before = repo_node_count(repo)
    live_before = live_node_ids(repo)
    timers, restore = install_turn_timers(repo.arena)
    started = time.perf_counter()
    answer = ""
    chat_node_id = None
    correction_result = None
    correction_wall_ms = 0.0
    plant_mode = None
    route_diag = None
    exc = None
    info: dict[str, Any] = {}
    try:
        # Fact/supersede plants: feed() COMPLETE (Leg-1). Probe/filler keep
        # production chat()→step() so the receipt measures real recall path.
        if event["kind"] in ("fact", "supersede"):
            answer, info, chat_node_id = plant_complete_turn(repo, event)
            plant_mode = info.get("plant_mode")
            if event["kind"] == "supersede":
                t0 = time.perf_counter()
                correction_result = repo.apply_memory_command(
                    event["correction_command"])
                correction_wall_ms = (time.perf_counter() - t0) * 1000.0
        else:
            if event["kind"] == "probe":
                source_turn = int(event["source_turn"])
                source_record = turn_records.get(source_turn, {})
                source_node = source_record.get(
                    "memory_node_id", source_record.get("chat_node_id"))
                route_diag = collect_route_diagnostics(
                    repo.arena, event["user"], source_node, top_k=5)
                # Fork A: widen probe mount from argmax to top-k at driver
                # call site (arena multi-mount picks list), not product patch.
                answer, info = probe_multimount_chat(
                    repo,
                    event["user"],
                    topk=int(args.topk),
                    ngen=int(args.ngen),
                )
                if route_diag is not None:
                    route_diag = dict(route_diag)
                    route_diag["mount_plan"] = info.get("mount_plan")
                    route_diag["mount_fitted"] = info.get("mount_fitted")
                    route_diag["mount_dropped_for_width"] = info.get(
                        "mount_dropped_for_width")
            else:
                answer, info = repo.runtime.chat(
                    event["user"],
                    ngen=int(args.ngen),
                    max_trips=int(args.max_trips),
                )
            last = repo.runtime.last_result
            new_nodes = list(getattr(last, "new_nodes", ()) or ())
            chat_node_id = int(new_nodes[0]) if new_nodes else None
    except Exception as err:  # receipt scripts should persist the failure row
        info = info or {}
        exc = repr(err)
    finally:
        restore()

    wall_ms = (time.perf_counter() - started) * 1000.0
    mounts_after = list(repo.arena.cur_mounts)
    nodes_after = repo_node_count(repo)
    live_after = live_node_ids(repo)
    transcript_row = {
        "turn": int(turn_idx),
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "user": event["user"],
        "assistant": answer,
        "chat_node_id": chat_node_id,
        "resumed": bool(resumed),
        "plant_mode": plant_mode,
    }
    transcript.append(transcript_row)
    append_jsonl(paths["transcript"], transcript_row)

    turn_records[int(turn_idx)] = {
        "chat_node_id": chat_node_id,
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "value": event.get("value") or event.get("expected"),
    }
    if correction_result and correction_result.get("node_id") is not None:
        turn_records[int(turn_idx)]["memory_node_id"] = int(correction_result["node_id"])

    eviction_check = None
    probe_score = None
    if event["kind"] == "probe":
        source_turn = int(event["source_turn"])
        source_record = turn_records.get(source_turn, {})
        source_node = source_record.get("memory_node_id", source_record.get("chat_node_id"))
        eviction_check = {
            "source_turn": source_turn,
            "live_turns": int(args.live_turns),
            "policy_evicted_before_probe": bool(source_turn < turn_idx - int(args.live_turns)),
            "source_node_id": source_node,
            "live_node_ids_before": live_before,
            "source_node_live_before": bool(source_node in live_before)
            if source_node is not None else None,
        }
        probe_score = score_probe(answer, event)
        # info["mounts"] from arena._attempt is 1-indexed; also keep the
        # 0-indexed seated set (cur_mounts / mount_fitted) for scorecards.
        mounted_ids = list(info.get("mount_fitted") or mounts_after)
        probe_score.update({
            "turn": int(turn_idx),
            "route_backend": repo.arena.last_route_backend,
            "mounts": info.get("mounts", []),
            "mounted_ids": [int(x) for x in mounted_ids],
            "mount_plan": info.get("mount_plan"),
            "eviction_check": eviction_check,
            "resumed": bool(resumed),
            "route_ranking": route_diag,
        })
        probe_rows.append(probe_score)

    row = {
        "schema": "grm_e2e_session_turn_v1",
        "turn": int(turn_idx),
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "resumed": bool(resumed),
        "route_backend": repo.arena.last_route_backend,
        "plant_mode": plant_mode,
        "mounts_before": mounts_before,
        "mounts_after": mounts_after,
        "mounts_changed": mounts_before != mounts_after,
        "live_node_ids_before": live_before,
        "live_node_ids_after": live_after,
        "live_window_tokens": int(info.get(
            "live_tokens", sum(n for _g, n in repo.arena.live_segs))),
        "repo_node_count": int(nodes_after),
        "repo_node_count_before": int(nodes_before),
        "new_node_count": int(nodes_after - nodes_before),
        "turn_wall_ms": wall_ms,
        "correction_wall_ms": correction_wall_ms,
        "vram": vram_snapshot(),
        "info": info,
        "error": exc,
        **timers.as_dict(),
    }
    if correction_result is not None:
        row["correction_result"] = correction_result
    if eviction_check is not None:
        row["eviction_check"] = eviction_check
    if route_diag is not None:
        row["route_ranking"] = route_diag
    if probe_score is not None:
        row["probe_score"] = probe_score
    append_jsonl(paths["instrumentation"], row)
    print(json.dumps({
        "turn": turn_idx,
        "kind": event["kind"],
        "backend": row["route_backend"],
        "plant_mode": plant_mode,
        "route_ms": round(row["route_wall_ms"], 3),
        "deposit_ms": round(row["deposit_wall_ms"], 3),
        "mount_ms": round(row["mount_wall_ms"], 3),
        "nodes": row["repo_node_count"],
        "probe_pass": probe_score.get("pass") if probe_score else None,
        "source_rank": (route_diag or {}).get("source_rank"),
        "mounted_ids": (probe_score or {}).get("mounted_ids"),
        "mount_plan": (probe_score or {}).get("mount_plan"),
        "error": exc,
    }), flush=True)
    if exc is not None:
        raise RuntimeError(f"turn {turn_idx} failed: {exc}")


def write_scorecard(paths: dict[str, Path], probe_rows: list[dict[str, Any]]) -> None:
    passed = sum(1 for row in probe_rows if row.get("pass"))
    write_json(paths["scorecard"], {
        "schema": "grm_e2e_probe_scorecard_v1",
        "passed": int(passed),
        "total": int(len(probe_rows)),
        "all_passed": bool(passed == len(probe_rows)),
        "probes": probe_rows,
    })


def maybe_restart(args: argparse.Namespace, session_dir: Path, paths: dict[str, Path],
                  repo: GraftRepository, turn_idx: int) -> None:
    print("checkpoint_restart=flush_now", flush=True)
    t0 = time.perf_counter()
    repo.flush_now()
    flush_ms = (time.perf_counter() - t0) * 1000.0
    write_json(paths["restart"], {
        "schema": "grm_e2e_restart_v1",
        "after_turn": int(turn_idx),
        "flush_wall_ms": flush_ms,
        "vram_before_exec": vram_snapshot(),
        "mode": args.mode,
    })
    try:
        repo.close()
    finally:
        del repo
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", args.mode,
        "--resume",
        "--session-dir", str(session_dir),
        "--model-dir", str(args.model_dir),
        "--native-lib", str(args.native_lib),
        "--ngen", str(args.ngen),
        "--max-trips", str(args.max_trips),
        "--live-turns", str(args.live_turns),
        "--arena-width", str(args.arena_width),
        "--max-live", str(args.max_live),
        "--topk", str(args.topk),
        "--restart-after", str(args.restart_after),
        "--skip-gpu-idle-check",
    ]
    os.execvpe(sys.executable, argv, os.environ.copy())


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.ngen is None:
        args.ngen = 24 if args.mode == "smoke" else 32
    if args.live_turns is None:
        args.live_turns = 1 if args.mode == "smoke" else 2
    script = script_for_mode(args.mode)
    if args.restart_after is None:
        args.restart_after = 5 if args.mode == "smoke" else len(script) // 2
    session_dir = args.session_dir or default_session_dir(args.mode)
    session_dir = session_dir.expanduser().resolve()
    paths = stage_paths(session_dir)
    if not args.resume and session_dir.exists():
        raise SystemExit(f"session dir already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    os.environ["GRM_GRAFT_STORAGE_BITS"] = "8"

    if not args.skip_gpu_idle_check:
        idle = wait_for_idle_gpu()
        if not idle["idle"]:
            write_json(paths["summary"], {
                "schema": "grm_e2e_session_summary_v1",
                "status": "blocked_gpu_busy",
                "gpu_idle_check": idle,
            })
            raise SystemExit("GPU stayed busy for 15 minutes")
    else:
        idle = {"idle": "skipped", "waited_s": 0, "checks": []}

    if not args.resume:
        write_json(paths["config"], {
            "schema": "grm_e2e_session_config_v1",
            "mode": args.mode,
            "session_dir": str(session_dir),
            "model_dir": str(args.model_dir),
            "native_lib": str(args.native_lib),
            "turns": len(script),
            "restart_after": int(args.restart_after),
            "live_turns": int(args.live_turns),
            "arena_width": int(args.arena_width),
            "topk": int(args.topk),
            "ngen": int(args.ngen),
            "max_trips": int(args.max_trips),
            "template_decision": (
                "ArenaCache prompt_template + stop_sequences hook; "
                "probe/filler use real GRMRuntime.chat()/ArenaCache.step(); "
                "fact/supersede plants use arena.feed() COMPLETE turns "
                "(scripted acceptance; value-bearing deposit, no free-gen)."
            ),
            "env": {
                "GRM_GQA_CUDA_ROUTE": os.environ["GRM_GQA_CUDA_ROUTE"],
                "GRM_GRAFT_STORAGE_BITS": os.environ["GRM_GRAFT_STORAGE_BITS"],
            },
            "gpu_idle_check": idle,
            "script": script,
        })

    transcript = read_jsonl(paths["transcript"])
    instrumentation = read_jsonl(paths["instrumentation"])
    probe_rows = []
    if paths["scorecard"].exists():
        probe_rows = json.loads(paths["scorecard"].read_text(encoding="utf-8")).get(
            "probes", [])
    completed = len(instrumentation)
    if args.resume:
        start_turn = completed
    else:
        start_turn = 0

    t_load = time.perf_counter()
    model, tokenizer, repo, model_info = load_model_and_repo(args, session_dir)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    refeed = []
    if args.resume:
        refeed = refeed_live_window(repo, transcript, int(args.live_turns))
        restart_payload = json.loads(paths["restart"].read_text(encoding="utf-8"))
        restart_payload.update({
            "resume_load_wall_ms": load_ms,
            "refeed": refeed,
            "vram_after_refeed": vram_snapshot(),
        })
        write_json(paths["restart"], restart_payload)

    turn_records: dict[int, dict[str, Any]] = {}
    for row in transcript:
        turn_records[int(row["turn"])] = {
            "chat_node_id": row.get("chat_node_id"),
            "kind": row.get("kind"),
            "fact_id": row.get("fact_id"),
        }
    for row in instrumentation:
        corr = row.get("correction_result") or {}
        if corr.get("node_id") is not None:
            turn_records.setdefault(int(row["turn"]), {})["memory_node_id"] = int(
                corr["node_id"])

    started = time.perf_counter()
    try:
        for turn_idx in range(start_turn, len(script)):
            run_turn(
                repo,
                script[turn_idx],
                turn_idx,
                paths=paths,
                transcript=transcript,
                turn_records=turn_records,
                probe_rows=probe_rows,
                args=args,
                resumed=bool(args.resume),
            )
            write_scorecard(paths, probe_rows)
            if (not args.resume) and turn_idx + 1 == int(args.restart_after):
                maybe_restart(args, session_dir, paths, repo, turn_idx)
    finally:
        try:
            repo.flush_now()
            repo.close()
        except Exception:
            pass
        del model
        del tokenizer
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    all_rows = read_jsonl(paths["instrumentation"])
    write_scorecard(paths, probe_rows)
    summary = {
        "schema": "grm_e2e_session_summary_v1",
        "status": "ok" if all(row.get("pass") for row in probe_rows) else "probe_failures",
        "mode": args.mode,
        "session_dir": str(session_dir),
        "turns_completed": len(all_rows),
        "turns_expected": len(script),
        "probes_total": len(probe_rows),
        "probes_passed": sum(1 for row in probe_rows if row.get("pass")),
        "supersession_probes": sum(1 for row in probe_rows if row.get("old_value")),
        "post_restart_probes": sum(1 for row in probe_rows if row.get("resumed")),
        "restart": json.loads(paths["restart"].read_text(encoding="utf-8"))
        if paths["restart"].exists() else None,
        "model_info": model_info,
        "load_wall_ms_last_process": load_ms,
        "run_wall_ms_this_process": (time.perf_counter() - started) * 1000.0,
        "sample_instrumentation": all_rows[-1] if all_rows else None,
    }
    write_json(paths["summary"], summary)
    print(f"summary={paths['summary']}", flush=True)
    print(json.dumps({
        "status": summary["status"],
        "turns": summary["turns_completed"],
        "probes": [summary["probes_passed"], summary["probes_total"]],
        "session_dir": str(session_dir),
    }), flush=True)
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
