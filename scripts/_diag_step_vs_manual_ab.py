#!/usr/bin/env python3
"""THROWAWAY diagnosis: ArenaCache.step()/_attempt generate vs Leg-1 manual
greedy loop on identical machinery, prompt bytes, and mounted graft.

Hard rules for this order:
  - diagnosis only (no product fix)
  - tracked tree must stay clean except artifacts/
  - GPU runs bounded <= 10 min each

Artifacts written under:
  artifacts/grm_e2e/step_vs_manual_ab_<stamp>/
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")

import numpy as np  # noqa: E402
import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core import kv_graft  # noqa: E402
from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs  # noqa: E402
from core.graft_arena import GQAArenaCache  # noqa: E402

SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)

SYS = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|>"
)
SYSTEM_PREFIX = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|><|start|>user<|message|>"
)
ASSISTANT_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"
HARMONY_STOPS = (
    "<|return|>", "<|end|>", "<|start|>user", "<|start|>assistant",
)

FACT_VALUE = "Vortex-3-Sierra"
FACT_USER = (
    f"Memory planting turn. fact cypher bridge. The current cypher bridge "
    f"value is {FACT_VALUE}. If asked later for cypher bridge, answer "
    f"{FACT_VALUE} only."
)
FACT_ASSIST = f"Understood. Stored cypher bridge = {FACT_VALUE}."
FILLER_USER = (
    "Session continuity filler smoke alpha. Acknowledge briefly. "
    "Do not change any stored fact values."
)
FILLER_ASSIST = "Acknowledged."
PROBE_USER = (
    "Recall probe. What is the current cypher bridge value? "
    "Reply with only the value."
)

NGEN = 16
TOPK_LOGITS = 5


def stamp_dir() -> Path:
    s = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = ROOT / "artifacts" / "grm_e2e" / f"step_vs_manual_ab_{s}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def harmony_turn(user: str, assistant: str | None) -> str:
    p = f"{SYSTEM_PREFIX}{user}{ASSISTANT_FINAL}"
    if assistant is None:
        return p
    return f"{p}{assistant}<|end|>"


def leg1_turn(user: str, assistant: str) -> str:
    return (
        f"<|start|>user<|message|>{user}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{assistant}<|end|>"
    )


def leg1_probe_prefix(user: str) -> str:
    return (
        f"<|start|>user<|message|>{user}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>"
    )


def cache_len_by_layer(arena) -> list[dict[str, Any]]:
    out = []
    if arena.caches is None:
        return out
    for li, cache in enumerate(arena.caches):
        k = cache[0]
        # PAYLOAD k dim=2 for GQA
        S = int(k.shape[2])
        layer = arena.m.layers[li]
        ltype = getattr(layer.self_attn, "layer_type", "?")
        out.append({
            "layer": li,
            "type": ltype,
            "S": S,
            "k_shape": list(k.shape),
            "v_shape": list(cache[1].shape),
        })
    return out


def summarize_cache(layers: list[dict[str, Any]]) -> dict[str, Any]:
    if not layers:
        return {"full_S": None, "sliding_S": None, "n_layers": 0}
    full = [x["S"] for x in layers if x["type"] == "full_attention"]
    slid = [x["S"] for x in layers if x["type"] == "sliding_attention"]
    return {
        "n_layers": len(layers),
        "full_S_unique": sorted(set(full)),
        "sliding_S_unique": sorted(set(slid)),
        "full_S0": full[0] if full else None,
        "sliding_S0": slid[0] if slid else None,
        "all_equal": len(set(x["S"] for x in layers)) == 1,
    }


def topk_from_row(row: np.ndarray, k: int = TOPK_LOGITS) -> list[dict[str, Any]]:
    idx = np.argpartition(row, -k)[-k:]
    idx = idx[np.argsort(-row[idx])]
    return [{"id": int(i), "logit": float(row[i])} for i in idx]


def instrumented_forward(arena, ids, dump: dict[str, Any], tok) -> np.ndarray:
    """One _forward with pre/post mechanical dumps."""
    ids = list(ids)
    pre_pos = int(arena.pos)
    pre_cache = cache_len_by_layer(arena)
    # live_shift as seen by attention modules
    shifts = []
    graft_seats = []
    inject_set = []
    for li, layer in enumerate(arena.m.layers):
        att = layer.self_attn
        shifts.append(getattr(att, "live_shift", None))
        graft_seats.append(getattr(att, "graft_seats", None))
        inject_set.append(att.inject_kv is not None)
    rope_start = pre_pos + int(arena.live_shift)

    row = arena._forward(ids)

    post_cache = cache_len_by_layer(arena)
    tok_id = int(row.argmax())
    top = topk_from_row(row)
    piece = {
        "input_ids": ids if len(ids) <= 8 else ids[:4] + ["..."] + ids[-2:],
        "input_n": len(ids),
        "pre_pos": pre_pos,
        "post_pos": int(arena.pos),
        "rope_start_expected": rope_start,
        "rope_end_expected": rope_start + len(ids) - 1,
        "live_shift": int(arena.live_shift),
        "n_sink": int(arena.n_sink),
        "cur_mount_n": int(arena.cur_mount_n),
        "width": int(arena.width),
        "live_shift_on_layers_unique": sorted({str(s) for s in shifts}),
        "graft_seats_unique": sorted({str(s) for s in graft_seats}),
        "inject_any": any(inject_set),
        "inject_count": int(sum(inject_set)),
        "pre_cache": summarize_cache(pre_cache),
        "post_cache": summarize_cache(post_cache),
        "chosen_id": tok_id,
        "chosen_text": tok.decode([tok_id]),
        "top5": [
            {**t, "text": tok.decode([t["id"]])} for t in top
        ],
    }
    # full layer dump only on first step (large)
    if dump.get("_save_full_layers"):
        piece["pre_cache_layers"] = pre_cache
        piece["post_cache_layers"] = post_cache
        dump["_save_full_layers"] = False
    dump.setdefault("steps", []).append(piece)
    return row


def manual_loop(arena, prompt_ids, ngen, tok, label: str) -> dict[str, Any]:
    """Leg-1 style: no early-stop, no extra last-token commit."""
    dump: dict[str, Any] = {
        "label": label,
        "mode": "manual_leg1",
        "ngen": ngen,
        "prompt_n": len(prompt_ids),
        "_save_full_layers": True,
        "steps": [],
    }
    for L in arena.m.layers:
        L.self_attn.live_shift = arena.live_shift
    row = instrumented_forward(arena, prompt_ids, dump, tok)
    out = [int(row.argmax())]
    for _ in range(ngen - 1):
        row = instrumented_forward(arena, [out[-1]], dump, tok)
        out.append(int(row.argmax()))
    dump["out_ids"] = out
    dump["out_text"] = tok.decode(out)
    dump["final_pos"] = int(arena.pos)
    dump["final_cache"] = summarize_cache(cache_len_by_layer(arena))
    return dump


def attempt_loop(arena, prompt_ids, ngen, tok, stops, label: str) -> dict[str, Any]:
    """Mirrors ArenaCache._attempt generate body after mounts are seated."""
    dump: dict[str, Any] = {
        "label": label,
        "mode": "attempt_step",
        "ngen": ngen,
        "prompt_n": len(prompt_ids),
        "stops": list(stops),
        "_save_full_layers": True,
        "steps": [],
        "stop_events": [],
    }
    for L in arena.m.layers:
        L.self_attn.live_shift = arena.live_shift
    row = instrumented_forward(arena, prompt_ids, dump, tok)
    # step/_attempt clears injection after first forward
    kv_graft.clear_injection(arena.m)
    dump["cleared_injection_after_prompt"] = True
    out = [int(row.argmax())]
    cached_out = 0
    stopped = False
    for step_i in range(ngen - 1):
        decoded = arena.decode(out)
        hit_stops = [s for s in stops if s in decoded]
        if hit_stops:
            stopped = True
            dump["stop_events"].append({
                "at_gen": step_i,
                "stops": hit_stops,
                "decoded": decoded,
                "out_ids": list(out),
            })
            break
        row = instrumented_forward(arena, [out[-1]], dump, tok)
        cached_out += 1
        out.append(int(row.argmax()))
    if not stopped and not any(s in arena.decode(out) for s in stops):
        instrumented_forward(arena, [out[-1]], dump, tok)
        cached_out += 1
        dump["committed_last_token"] = True
    else:
        dump["committed_last_token"] = False
    txt = arena.decode(out)
    for stop in stops:
        if stop in txt:
            txt = txt.split(stop)[0]
    dump["out_ids"] = out
    dump["out_text_raw"] = arena.decode(out)
    dump["out_text"] = txt.strip()
    dump["cached_out"] = cached_out
    dump["stopped"] = stopped
    dump["final_pos"] = int(arena.pos)
    dump["final_cache"] = summarize_cache(cache_len_by_layer(arena))
    return dump


def clone_tensor(t):
    """Host-roundtrip copy (tensor_cuda has no .clone())."""
    arr = np.ascontiguousarray(t.numpy())
    # preserve compute dtype used by the model
    from core.mistral7b_tc import BlockTC
    return tc.tensor(arr, dtype="float32").astype(BlockTC.COMPUTE_DTYPE)


def clone_caches(caches):
    if caches is None:
        return None
    return [tuple(clone_tensor(t) for t in cache) for cache in caches]


def restore_state(arena, snap):
    arena.caches = clone_caches(snap["caches"])
    arena.pos = int(snap["pos"])
    arena.live_segs = list(snap["live_segs"])
    arena.cur_mounts = list(snap["cur_mounts"])
    arena.cur_mount_n = int(snap["cur_mount_n"])
    for L in arena.m.layers:
        L.self_attn.live_shift = arena.live_shift
        L.self_attn.inject_kv = None
        L.self_attn.graft_seats = 0


def make_arena(model, tok, *, sink_text: str, arena_width: int = 96,
               live_turns: int = 1, prompt_template=None, stop_sequences=None):
    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    encode = lambda t: tok.encode(t, add_special_tokens=False)
    decode = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)
    return GQAArenaCache(
        model,
        encode=encode,
        decode=decode,
        sink_text=sink_text,
        arena_width=arena_width,
        route_layer=int(dialect["route_layer"]),
        topk=1,
        live_turns=live_turns,
        max_live=1024,
        cache_deposits=True,
        prompt_template=prompt_template,
        stop_sequences=stop_sequences,
    )


def setup_leg1_mounted(model, tok) -> tuple[Any, dict[str, Any]]:
    """One fact deposited+evicted, fact graft mounted, ready for probe gen."""
    arena = make_arena(model, tok, sink_text=SYS, arena_width=96, live_turns=1)
    # plant with feed (complete turns — Leg-1; isolates generate-path delta)
    arena.feed(leg1_turn(FACT_USER, FACT_ASSIST), deposit=True)
    g_fact = 0
    arena.feed(leg1_turn(FILLER_USER, FILLER_ASSIST), deposit=True)
    live_idx = {g for g, _ in arena.live_segs if g is not None}
    assert g_fact not in live_idx, f"fact still live: {arena.live_segs}"
    ranking = arena.route(PROBE_USER, exclude=live_idx, limit=1)
    picks = sorted(ranking)
    # force-mount fact if route missed (diagnosis still wants same machinery)
    if g_fact not in picks:
        picks = [g_fact]
    arena._ensure_h(picks)
    arena.swap(picks)
    meta = {
        "n_sink": arena.n_sink,
        "width": arena.width,
        "live_shift": arena.live_shift,
        "pos": arena.pos,
        "cur_mounts": list(arena.cur_mounts),
        "cur_mount_n": arena.cur_mount_n,
        "live_segs": list(arena.live_segs),
        "ranking": ranking,
        "picks": picks,
        "picks_forced": ranking != picks,
        "graft0_ntok": arena.grafts[0]["ntok"],
        "graft0_text_head": arena.grafts[0]["text"][:120],
        "cache": summarize_cache(cache_len_by_layer(arena)),
    }
    return arena, meta


def first_divergence(a_steps, b_steps) -> dict[str, Any] | None:
    n = min(len(a_steps), b_steps and len(b_steps) or 0)
    for i in range(n):
        a, b = a_steps[i], b_steps[i]
        keys = (
            "pre_pos", "post_pos", "rope_start_expected", "chosen_id",
            "pre_cache", "post_cache", "live_shift", "cur_mount_n",
        )
        diffs = {}
        for k in keys:
            if a.get(k) != b.get(k):
                diffs[k] = {"A": a.get(k), "B": b.get(k)}
        # top5 id sequence
        a_top = [t["id"] for t in a.get("top5", [])]
        b_top = [t["id"] for t in b.get("top5", [])]
        if a_top != b_top:
            diffs["top5_ids"] = {"A": a_top, "B": b_top}
        if diffs:
            return {
                "step": i,
                "diffs": diffs,
                "A_chosen": {"id": a["chosen_id"], "text": a["chosen_text"]},
                "B_chosen": {"id": b["chosen_id"], "text": b["chosen_text"]},
                "A_pre_pos": a["pre_pos"],
                "B_pre_pos": b["pre_pos"],
                "A_rope_start": a["rope_start_expected"],
                "B_rope_start": b["rope_start_expected"],
                "A_cache": a["pre_cache"],
                "B_cache": b["pre_cache"],
            }
    if len(a_steps) != len(b_steps):
        return {
            "step": n,
            "diffs": {"n_steps": {"A": len(a_steps), "B": len(b_steps)}},
        }
    return None


def classify(div: dict[str, Any] | None, a_dump, b_dump) -> dict[str, Any]:
    if div is None:
        return {
            "class": "NO_DIVERGENCE",
            "note": (
                "Both loops produced identical pos/cache/token streams. "
                "Mechanical hypothesis (a)-(d) for the pure generate loop is "
                "refuted under this setup; residual defect is session-scale."
            ),
            "candidates": [],
        }
    diffs = div.get("diffs", {})
    cands = []
    if any(k in diffs for k in ("pre_pos", "post_pos", "rope_start_expected",
                                "pre_cache", "post_cache", "cur_mount_n")):
        cands.append("a_kv_position_misalignment")
    if a_dump.get("mode") != b_dump.get("mode"):
        # structural mode delta always present in labels; check stop/commit
        if b_dump.get("stop_events") or b_dump.get("committed_last_token"):
            cands.append("c_stop_or_commit_slicing")
    if "chosen_id" in diffs and not any(
            k in diffs for k in ("pre_pos", "rope_start_expected", "pre_cache")):
        # same mechanical inputs, different token → sampling/config or hidden state
        cands.append("d_sampling_or_hidden_state_delta")
    if not cands:
        cands.append("e_unknown_token_divergence")
    return {
        "class": cands[0],
        "candidates": cands,
        "first_div_step": div.get("step"),
        "note": "See divergence receipt for numeric mismatch.",
    }


def run_ab_on_mounted(model, tok, out_dir: Path) -> dict[str, Any]:
    """Core A/B: rebuild identical mounted state per arm (no cache clone)."""
    # probe prompts (session vs Leg-1 historical)
    prompt_leg1 = leg1_probe_prefix(PROBE_USER)
    prompt_step = harmony_turn(PROBE_USER, None)
    # temp encode via a throwaway arena-less path
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    ids_leg1 = enc(prompt_leg1)
    ids_step = enc(prompt_step)
    prompts = {
        "leg1_prefix": {
            "text": prompt_leg1,
            "sha256": sha256_bytes(prompt_leg1.encode("utf-8")),
            "n_ids": len(ids_leg1),
            "ids_head": ids_leg1[:16],
        },
        "step_harmony_turn": {
            "text": prompt_step,
            "sha256": sha256_bytes(prompt_step.encode("utf-8")),
            "n_ids": len(ids_step),
            "ids_head": ids_step[:16],
        },
        "byte_equal": prompt_leg1 == prompt_step,
        "id_equal": ids_leg1 == ids_step,
    }
    compare_ids = ids_step
    compare_label = "step_harmony_prompt"
    write_json(out_dir / "prompts.json", prompts)

    # --- A: manual on step-format prompt ---
    arena, meta = setup_leg1_mounted(model, tok)
    write_json(out_dir / "setup_meta.json", meta)
    t0 = time.perf_counter()
    dump_a = manual_loop(arena, compare_ids, NGEN, tok, "A_manual")
    dump_a["wall_s"] = time.perf_counter() - t0
    dump_a["pre_gen_pos"] = meta["pos"]
    dump_a["pre_gen_cache"] = meta["cache"]
    write_json(out_dir / "dump_A_manual.json", dump_a)
    del arena
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()

    # --- B: attempt on same prompt ---
    arena, meta_b = setup_leg1_mounted(model, tok)
    t0 = time.perf_counter()
    dump_b = attempt_loop(arena, compare_ids, NGEN, tok, HARMONY_STOPS, "B_attempt")
    dump_b["wall_s"] = time.perf_counter() - t0
    dump_b["setup_pos"] = meta_b["pos"]
    dump_b["setup_cache"] = meta_b["cache"]
    write_json(out_dir / "dump_B_attempt.json", dump_b)
    del arena
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()

    # --- C: manual on leg1 prefix if different ---
    dump_c = None
    if ids_leg1 != ids_step:
        arena, _ = setup_leg1_mounted(model, tok)
        t0 = time.perf_counter()
        dump_c = manual_loop(arena, ids_leg1, NGEN, tok, "C_manual_leg1_prefix")
        dump_c["wall_s"] = time.perf_counter() - t0
        write_json(out_dir / "dump_C_manual_leg1_prefix.json", dump_c)
        del arena
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    # --- D: actual step() end-to-end ---
    arena2, meta2 = setup_leg1_mounted(model, tok)
    arena2.prompt_template = harmony_turn
    arena2.stop_sequences = HARMONY_STOPS
    t0 = time.perf_counter()
    ans_step, info_step = arena2.step(PROBE_USER, ngen=NGEN, deposit=False,
                                      max_trips=0)
    wall_step = time.perf_counter() - t0
    dump_d = {
        "label": "D_step_e2e",
        "mode": "step",
        "answer": ans_step,
        "info": info_step,
        "wall_s": wall_step,
        "final_pos": int(arena2.pos),
        "final_cache": summarize_cache(cache_len_by_layer(arena2)),
        "mounts": info_step.get("mounts"),
        "setup": meta2,
    }
    write_json(out_dir / "dump_D_step_e2e.json", dump_d)
    del arena2
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()

    # setup mechanical equality check (rebuilds are independent but should match)
    setup_equal = (
        meta.get("pos") == meta_b.get("pos")
        and meta.get("cache") == meta_b.get("cache")
        and meta.get("cur_mount_n") == meta_b.get("cur_mount_n")
    )

    div_ab = first_divergence(dump_a["steps"], dump_b["steps"])
    cls_ab = classify(div_ab, dump_a, dump_b)
    div_ac = None
    if dump_c is not None:
        div_ac = first_divergence(dump_a["steps"], dump_c["steps"])

    receipt = {
        "schema": "grm_e2e_step_vs_manual_ab_v1",
        "compare_prompt": compare_label,
        "prompt_sha256": prompts["step_harmony_turn"]["sha256"],
        "prompts": prompts,
        "setup": meta,
        "setup_rebuild_equal_A_B": setup_equal,
        "A_text": dump_a["out_text"],
        "B_text": dump_b["out_text"],
        "C_text": None if dump_c is None else dump_c["out_text"],
        "D_step_text": ans_step,
        "A_ids": dump_a["out_ids"],
        "B_ids": dump_b["out_ids"],
        "divergence_A_vs_B": div_ab,
        "classification_A_vs_B": cls_ab,
        "divergence_A_vs_C_prompt_delta": div_ac,
        "value_in_A": FACT_VALUE.lower() in dump_a["out_text"].lower(),
        "value_in_B": FACT_VALUE.lower() in dump_b["out_text"].lower(),
        "value_in_C": (
            None if dump_c is None
            else FACT_VALUE.lower() in dump_c["out_text"].lower()
        ),
        "value_in_D": FACT_VALUE.lower() in ans_step.lower(),
        "walls": {
            "A": dump_a["wall_s"],
            "B": dump_b["wall_s"],
            "C": None if dump_c is None else dump_c["wall_s"],
            "D": wall_step,
        },
    }
    write_json(out_dir / "divergence_receipt.json", receipt)
    return receipt


def run_cold_bootstrap_ab(model, tok, out_dir: Path) -> dict[str, Any]:
    """Cold arena, no prior live turns: generate path only (bootstrap inject)."""
    prompt = harmony_turn(PROBE_USER, None)
    ids = None

    # A
    arena = make_arena(model, tok, sink_text=SYS, arena_width=96, live_turns=1,
                       prompt_template=harmony_turn, stop_sequences=HARMONY_STOPS)
    # bootstrap inject sink only (no mounts) — mirror _attempt caches is None
    mounts = [{"h": arena.sink_h}]
    _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
    inj = []
    for li in range(len(arena.m.layers)):
        inj.append({key: np.concatenate([_np(g["h"][li][key]) for g in mounts],
                                        axis=dim)
                    for key, dim in arena.PAYLOAD})
    arena._set_injection_host(inj)
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    ids = arena.encode(prompt)
    # snapshot post-inject pre-forward is empty caches; just run A then rebuild

    t0 = time.perf_counter()
    dump_a = manual_loop(arena, ids, NGEN, tok, "cold_A_manual")
    dump_a["wall_s"] = time.perf_counter() - t0
    write_json(out_dir / "cold_dump_A_manual.json", dump_a)

    # B
    arena = make_arena(model, tok, sink_text=SYS, arena_width=96, live_turns=1,
                       prompt_template=harmony_turn, stop_sequences=HARMONY_STOPS)
    mounts = [{"h": arena.sink_h}]
    inj = []
    for li in range(len(arena.m.layers)):
        inj.append({key: np.concatenate([_np(g["h"][li][key]) for g in mounts],
                                        axis=dim)
                    for key, dim in arena.PAYLOAD})
    arena._set_injection_host(inj)
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    ids = arena.encode(prompt)
    t0 = time.perf_counter()
    dump_b = attempt_loop(arena, ids, NGEN, tok, HARMONY_STOPS, "cold_B_attempt")
    dump_b["wall_s"] = time.perf_counter() - t0
    write_json(out_dir / "cold_dump_B_attempt.json", dump_b)

    # C: true step() cold
    arena = make_arena(model, tok, sink_text=SYS, arena_width=96, live_turns=1,
                       prompt_template=harmony_turn, stop_sequences=HARMONY_STOPS)
    t0 = time.perf_counter()
    ans, info = arena.step(PROBE_USER, ngen=NGEN, deposit=False, max_trips=0)
    wall = time.perf_counter() - t0

    # D: pure model, no arena (position_offset=0, no live_shift)
    for L in model.layers:
        L.self_attn.live_shift = None
        L.self_attn.inject_kv = None
        L.self_attn.graft_seats = 0
    prompt_ids = np.array([ids], dtype=np.int64)
    t0 = time.perf_counter()
    pure_out = []
    with tc.no_grad():
        lg, caches = model(prompt_ids, last_token_only=True)
        pure_out.append(int(lg.numpy()[0, -1].argmax()))
        pos = int(prompt_ids.shape[1])
        for _ in range(NGEN - 1):
            lg, caches = model(
                np.array([[pure_out[-1]]], dtype=np.int64),
                kv_caches=caches, position_offset=pos, last_token_only=True,
            )
            pos += 1
            pure_out.append(int(lg.numpy()[0, -1].argmax()))
    pure_wall = time.perf_counter() - t0
    pure_text = tok.decode(pure_out)

    div = first_divergence(dump_a["steps"], dump_b["steps"])
    receipt = {
        "schema": "grm_e2e_step_vs_manual_cold_ab_v1",
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "prompt_n_ids": len(ids),
        "A_text": dump_a["out_text"],
        "B_text": dump_b["out_text"],
        "C_step_text": ans,
        "C_info": info,
        "D_pure_model_text": pure_text,
        "D_pure_model_ids": pure_out,
        "divergence_A_vs_B": div,
        "classification_A_vs_B": classify(div, dump_a, dump_b),
        "A_ids": dump_a["out_ids"],
        "B_ids": dump_b["out_ids"],
        "walls": {"A": dump_a["wall_s"], "B": dump_b["wall_s"],
                  "C": wall, "D_pure": pure_wall},
    }
    write_json(out_dir / "cold_divergence_receipt.json", receipt)
    return receipt


def main() -> int:
    out_dir = stamp_dir()
    print(f"artifacts: {out_dir}", flush=True)
    os.environ.setdefault("GRM_GQA_CUDA_ROUTE", "1")
    os.environ.setdefault("GRM_GRAFT_STORAGE_BITS", "8")

    t_all = time.perf_counter()
    print("loading model...", flush=True)
    model, model_info = GptOss20B_TC.from_pretrained(SNAPSHOT)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    write_json(out_dir / "model_info.json", model_info)

    print("RUN1: mounted A/B (<=10min bound)", flush=True)
    t0 = time.perf_counter()
    r1 = run_ab_on_mounted(model, tok, out_dir)
    r1_wall = time.perf_counter() - t0
    print(json.dumps({
        "run": "mounted",
        "wall_s": round(r1_wall, 2),
        "A": r1["A_text"][:80],
        "B": r1["B_text"][:80],
        "D": r1["D_step_text"][:80],
        "div": r1["divergence_A_vs_B"],
        "class": r1["classification_A_vs_B"],
        "value_hits": {
            "A": r1["value_in_A"], "B": r1["value_in_B"],
            "C": r1["value_in_C"], "D": r1["value_in_D"],
        },
    }, indent=2, default=str), flush=True)
    if r1_wall > 600:
        print("WARN: mounted run exceeded 10min bound", flush=True)

    print("RUN2: cold bootstrap A/B", flush=True)
    t0 = time.perf_counter()
    r2 = run_cold_bootstrap_ab(model, tok, out_dir)
    r2_wall = time.perf_counter() - t0
    print(json.dumps({
        "run": "cold",
        "wall_s": round(r2_wall, 2),
        "A": r2["A_text"][:80],
        "B": r2["B_text"][:80],
        "C": r2["C_step_text"][:80],
        "D_pure": r2["D_pure_model_text"][:80],
        "div": r2["divergence_A_vs_B"],
        "class": r2["classification_A_vs_B"],
    }, indent=2, default=str), flush=True)

    summary = {
        "schema": "grm_e2e_step_vs_manual_ab_summary_v1",
        "out_dir": str(out_dir),
        "total_wall_s": time.perf_counter() - t_all,
        "mounted": r1,
        "cold": r2,
        "mounted_wall_s": r1_wall,
        "cold_wall_s": r2_wall,
    }
    write_json(out_dir / "summary.json", summary)
    print(f"TOTAL wall_s: {summary['total_wall_s']:.1f}", flush=True)
    print(f"DONE -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
