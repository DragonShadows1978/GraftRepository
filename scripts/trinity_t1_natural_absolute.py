#!/usr/bin/env python3
"""T1 NoPE-graft absolute verdict via natural-continuation readout.

No chat template is used anywhere in this harness. It reuses the Trinity
GQAArenaCache adapter from scripts/trinity_nope_graft_width_sweep.py and runs
the registered stop rail first: if the value cannot be read from plain live
context, the graft grid is not run.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/mnt/ForgeRealm/models/trinity-nano")
REF_DIR = Path("/mnt/ForgeRealm/Project-Tensor/artifacts/trinity_nano/reference_capture")
OUT_ROOT = ROOT / "artifacts" / "trinity_nope_graft"
GPU_WALL_RAIL_S = 600.0
DRIFT_STEPS = 8
DRIFT_STOP_MAX_ABS = 1.0

sys.path.insert(0, str(ROOT / "scripts"))

import trinity_nope_graft_width_sweep as W  # noqa: E402

tc = W.tc
kv_graft = W.kv_graft

FACT_VALUE = "Vortex-3-Sierra"
PLANT_TEXT = (
    f"For the record, the vault access code is {FACT_VALUE}. "
    "Keep it safe.\n"
)
FILLER_TEXT = (
    "The next notebook paragraph describes a quiet maintenance walk through "
    "the archive corridor. The lights were checked, the shelves were dusted, "
    "and the operator wrote down that routine observations should continue "
    "without changing any earlier record.\n"
)
READOUT_PROMPT = "As noted earlier, the vault access code is"
SINK_TEXT = "Notebook context:\n"
WIDTHS = (96, 768)
NGEN = 16


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def bytes_gib(n: int | float | None) -> float | None:
    if n is None:
        return None
    return round(float(n) / float(1024 ** 3), 4)


def nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def smi_used_mib(row: str | None) -> int | None:
    if not row:
        return None
    try:
        return int(row.splitlines()[0].split(",")[1].strip())
    except Exception:
        return None


class VramSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample("poll")
            self._stop.wait(self.interval_s)

    def sample(self, tag: str) -> None:
        smi = nvidia_smi()
        self.samples.append({
            "t": time.time(),
            "tag": str(tag),
            "smi": smi,
            "used_mib": smi_used_mib(smi),
        })

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        used = [s["used_mib"] for s in self.samples if s.get("used_mib") is not None]
        peak = max(used) if used else None
        return {
            "evidence_class": "nvidia-smi sampled process-adjacent VRAM, not allocator peak",
            "sample_interval_s": self.interval_s,
            "num_samples": len(self.samples),
            "max_used_mib": peak,
            "max_used_gib": bytes_gib(peak * 1024 * 1024) if peak is not None else None,
            "first": self.samples[0] if self.samples else None,
            "last": self.samples[-1] if self.samples else None,
        }


def code_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def value_recovered(text: str, value: str = FACT_VALUE) -> bool:
    return code_norm(value) in code_norm(text)


def gen_class(text: str, ids: list[int]) -> str:
    if value_recovered(text):
        return "value_hit"
    if W.is_clean_english(text):
        return "clean_english"
    return W.collapse_signature(text, ids)


def clean_text(tok, ids: list[int]) -> tuple[str, str]:
    raw = tok.decode(ids, clean_up_tokenization_spaces=False)
    text = raw
    for stop in W.STOPS:
        if stop in text:
            text = text.split(stop)[0]
    return raw, text.strip()


def token_texts(tok, ids: list[int]) -> list[str]:
    return [tok.decode([int(i)], clean_up_tokenization_spaces=False) for i in ids]


def clear_dials(adapter) -> None:
    for layer in adapter.layers:
        att = layer.self_attn
        att.live_shift = None
        att.inject_kv = None
        att.graft_seats = 0


def direct_greedy(
    adapter,
    tok,
    prompt_text: str,
    ngen: int,
    label: str,
    *,
    weight_mode: str,
) -> dict[str, Any]:
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    t0 = time.perf_counter()
    out: list[int] = []
    with tc.no_grad():
        clear_dials(adapter)
        logits, caches = adapter(
            np.array([prompt_ids], dtype=np.int64),
            last_token_only=True,
        )
        for step in range(int(ngen)):
            nid = int(np.argmax(logits.float().numpy()[0, -1]))
            out.append(nid)
            if tok.eos_token_id is not None and nid == int(tok.eos_token_id):
                break
            logits, caches = adapter(
                np.array([[nid]], dtype=np.int64),
                kv_caches=caches,
                position_offset=len(prompt_ids) + step,
                last_token_only=True,
            )
    raw, text = clean_text(tok, out)
    row = {
        "label": label,
        "mode": f"{weight_mode}_fp32_resident_no_arena",
        "chat_template_used": False,
        "prompt_text": prompt_text,
        "prompt_ntok": len(prompt_ids),
        "generated_ids": out,
        "generated_token_texts": token_texts(tok, out),
        "generated_text_raw": raw,
        "generated_text": text,
        "n_steps": len(out),
        "value_recovered": value_recovered(text),
        "class": gen_class(text, out),
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    try:
        del caches, logits
    except UnboundLocalError:
        pass
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return row


def load_p1_reference(ref_dir: Path) -> dict[str, Any]:
    path = Path(ref_dir) / "probe_00_plain_short.npz"
    z = np.load(path)
    return {
        "path": str(path),
        "input_ids": z["input_ids"].astype(np.int64),
        "generated_ids": z["generated_ids"].astype(np.int64),
        "logits": z["logits"].astype(np.float32),
        "top5_ids": z["top5_ids"].astype(np.int64),
    }


def top5_ids(row: np.ndarray) -> np.ndarray:
    return np.argsort(-row)[:5].astype(np.int64)


def run_p1_drift(
    adapter,
    ref_dir: Path,
    *,
    steps: int,
    stop_max_abs: float,
) -> dict[str, Any]:
    ref = load_p1_reference(ref_dir)
    input_ids = ref["input_ids"].reshape(1, -1)
    generated_ids = ref["generated_ids"].reshape(-1)
    current = input_ids
    caches = None
    rows = []
    prompt_len = int(input_ids.shape[1])
    t0 = time.perf_counter()
    with tc.no_grad():
        clear_dials(adapter)
        for step in range(min(int(steps), int(ref["logits"].shape[0]))):
            logits, caches = adapter(
                current,
                kv_caches=caches,
                position_offset=0 if step == 0 else prompt_len + step - 1,
                last_token_only=True,
            )
            got = logits.float().numpy()[0, -1].astype(np.float32)
            want = ref["logits"][step].astype(np.float32, copy=False)
            got_top = top5_ids(got)
            ref_top = ref["top5_ids"][step].astype(np.int64, copy=False)
            overlap = sorted({int(x) for x in got_top.tolist()} & {int(x) for x in ref_top.tolist()})
            rows.append({
                "step": int(step),
                "max_abs_delta_logit": float(np.max(np.abs(got - want))),
                "mean_abs_delta_logit": float(np.mean(np.abs(got - want))),
                "int8_top5_ids": [int(x) for x in got_top.tolist()],
                "bf16_top5_ids": [int(x) for x in ref_top.tolist()],
                "top5_exact": bool(np.array_equal(got_top, ref_top)),
                "top5_overlap_count": int(len(overlap)),
                "top5_overlap_ids": overlap,
                "argmax_id": int(got_top[0]),
                "bf16_argmax_id": int(ref_top[0]),
            })
            feed_id = int(generated_ids[step])
            current = np.asarray([[feed_id]], dtype=np.int64)
            del logits
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
    max_abs = max((r["max_abs_delta_logit"] for r in rows), default=None)
    exact = int(sum(1 for r in rows if r["top5_exact"]))
    return {
        "evidence_class": "teacher-forced INT8-resident logits vs saved BF16 HF P1 logits",
        "reference_path": ref["path"],
        "steps": rows,
        "step_count": len(rows),
        "max_abs_delta_logit": max_abs,
        "top5_exact_steps": exact,
        "top5_total_steps": len(rows),
        "top5_overlap_total": int(sum(r["top5_overlap_count"] for r in rows)),
        "stop_max_abs_delta_logit": float(stop_max_abs),
        "multi_logit_drift_stop": bool(max_abs is not None and max_abs > float(stop_max_abs)),
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def conversion_receipt(model_dir: Path, model_info: dict[str, Any]) -> dict[str, Any]:
    shard_bytes = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    qbytes = int(model_info.get("quantized_linear_bytes") or 0)
    source_bytes = int(model_info.get("source_bf16_linear_bytes") or 0)
    return {
        "evidence_class": "loader conversion accounting from TensorCUDA model_info plus safetensors stat",
        "source_safetensors_on_disk_bytes": int(shard_bytes),
        "source_safetensors_on_disk_gib": bytes_gib(shard_bytes),
        "source_bf16_linear_bytes_est": source_bytes,
        "source_bf16_linear_gib_est": bytes_gib(source_bytes),
        "resident_quantized_linear_bytes": qbytes,
        "resident_quantized_linear_gib": bytes_gib(qbytes),
        "embedding_host_bytes": int(model_info.get("embedding_host_bytes") or 0),
        "lm_head_loaded": bool(model_info.get("lm_head_loaded")),
        "model_info_weight_mode": model_info.get("weight_mode"),
        "note": "INT8 weights are quantized on load; no separate 6GB weight shard was materialized.",
    }


def make_arena(adapter, tok, width: int) -> W.TrinityGQAArenaCache:
    encode = lambda t: tok.encode(t, add_special_tokens=False)
    decode = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)
    return W.TrinityGQAArenaCache(
        adapter,
        encode=encode,
        decode=decode,
        sink_text=SINK_TEXT,
        arena_width=int(width),
        route_layer=0,
        topk=1,
        live_turns=1,
        max_live=512,
        cache_deposits=True,
        stop_sequences=W.STOPS,
    )


def cache_summary(arena) -> dict[str, Any]:
    return W.cache_layer_summary(arena)


def arena_greedy(arena, tok, prompt_text: str, ngen: int) -> dict[str, Any]:
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    t0 = time.perf_counter()
    out: list[int] = []
    steps = []
    row = arena._forward(prompt_ids)
    kv_graft.clear_injection(arena.m)
    out.append(int(row.argmax()))
    steps.append({
        "tag": "prompt",
        "chosen": out[-1],
        "text": tok.decode([out[-1]], clean_up_tokenization_spaces=False),
        "arena_pos_after_prompt": int(arena.pos),
        "rope0_expected": int(arena.live_shift) + (int(arena.pos) - len(prompt_ids)),
    })
    for step in range(int(ngen) - 1):
        if tok.eos_token_id is not None and out[-1] == int(tok.eos_token_id):
            break
        row = arena._forward([out[-1]])
        out.append(int(row.argmax()))
        if step < 3:
            steps.append({
                "tag": f"g{step}",
                "chosen": out[-1],
                "text": tok.decode([out[-1]], clean_up_tokenization_spaces=False),
                "arena_pos": int(arena.pos),
            })
    raw, text = clean_text(tok, out)
    return {
        "prompt_text": prompt_text,
        "prompt_ntok": len(prompt_ids),
        "generated_ids": out,
        "generated_token_texts": token_texts(tok, out),
        "generated_text_raw": raw,
        "generated_text": text,
        "n_steps": len(out),
        "value_recovered": value_recovered(text),
        "class": gen_class(text, out),
        "steps_head": steps,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def run_grid_case(adapter, tok, *, width: int, mount: bool, ngen: int) -> dict[str, Any]:
    label = f"w{width}_{'mount' if mount else 'control'}"
    t0 = time.perf_counter()
    arena = make_arena(adapter, tok, width)
    row: dict[str, Any] = {
        "label": label,
        "width": int(width),
        "mount": bool(mount),
        "sink_text": SINK_TEXT,
        "sink_ntok": int(arena.n_sink),
        "live_shift": int(arena.live_shift),
        "chat_template_used": False,
    }
    arena.feed(PLANT_TEXT, deposit=True)
    fact_gidx = 0
    row["plant_text"] = PLANT_TEXT
    row["plant_ntok"] = int(arena.grafts[fact_gidx]["ntok"])
    row["cache_after_plant"] = cache_summary(arena)

    arena.feed(FILLER_TEXT, deposit=True)
    live_idx = {g for g, _ in arena.live_segs if g is not None}
    row["filler_text"] = FILLER_TEXT
    row["live_segs_after_evict"] = [
        (None if g is None else int(g), int(n)) for g, n in arena.live_segs
    ]
    row["fact_evicted"] = fact_gidx not in live_idx
    row["cache_after_evict"] = cache_summary(arena)

    if mount:
        ranking = arena.route(READOUT_PROMPT, exclude=live_idx, limit=1)
        picks = list(ranking) if ranking else []
        picks_forced = False
        if fact_gidx not in picks:
            picks = [fact_gidx]
            picks_forced = True
        arena.swap(picks)
        row["ranking"] = [int(x) for x in ranking]
        row["picks"] = [int(x) for x in picks]
        row["picks_forced"] = picks_forced
    else:
        arena.swap([])
        row["ranking"] = []
        row["picks"] = []
        row["picks_forced"] = False

    row["cur_mounts"] = [int(x) for x in arena.cur_mounts]
    row["cur_mount_n"] = int(arena.cur_mount_n)
    row["cache_pre_gen"] = cache_summary(arena)
    row["generation"] = arena_greedy(arena, tok, READOUT_PROMPT, ngen)
    row["cache_post_gen"] = cache_summary(arena)
    row["wall_s"] = round(time.perf_counter() - t0, 3)

    arena.reset_live_cache()
    del arena
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return row


def decide(sanity: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not sanity["value_recovered"]:
        return {
            "verdict": "SANITY_FAILED_STOP",
            "reason": (
                "Plain live-context natural continuation did not recover the "
                f"code; stop rail hit before graft grid. Quote: "
                f"{sanity['generated_text']!r}"
            ),
            "sanity_passed": False,
            "grid_ran": False,
        }

    by = {c["label"]: c for c in cases}
    def hit(label: str) -> bool:
        return bool((by.get(label, {}).get("generation") or {}).get("value_recovered"))

    m96 = hit("w96_mount")
    m768 = hit("w768_mount")
    c96 = hit("w96_control")
    c768 = hit("w768_control")

    if m768 and not c96 and not c768:
        verdict = "T1_ABSOLUTE_CONFIRMED"
        reason = (
            "Value recovered from the width-768 mounted graft while both "
            "no-mount controls missed; the graft, not live context, carried it."
        )
    elif m96 and not m768:
        verdict = "T1_REFUTED_HOLE_EFFECT_ON_NOPE"
        reason = (
            "Value recovered at width 96 but not width 768; this is a "
            "width-sensitive hole effect on the Trinity NoPE graft path."
        )
    elif not m96 and not m768:
        verdict = "READOUT_ROUTE_SEAM_NOT_HOLE"
        reason = (
            "Sanity passed, but neither mounted width recovered the value; "
            "this points at readout/route/mount seam rather than a NoPE hole."
        )
    elif c96 or c768:
        verdict = "CONTROL_LEAKAGE_INCONCLUSIVE"
        reason = (
            "A no-mount control recovered the value, so the receipt cannot "
            "prove the graft carried the fact."
        )
    else:
        verdict = "INCONCLUSIVE_PATTERN"
        reason = (
            "Observed recovery pattern was outside the registered decision "
            "forks; inspect per-arm receipts."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "sanity_passed": True,
        "grid_ran": True,
        "hits": {
            "w96_mount": m96,
            "w96_control": c96,
            "w768_mount": m768,
            "w768_control": c768,
        },
    }


def write_summary(out_dir: Path, receipt: dict[str, Any]) -> None:
    verdict = receipt["verdict"]
    sanity = receipt.get("sanity")
    table = receipt.get("grid_table") or []
    lines = [
        f"# T1 Natural Absolute — {verdict['verdict']}",
        "",
        f"**Reason:** {verdict['reason']}",
    ]
    conv = receipt.get("conversion_receipt")
    if conv:
        lines.extend([
            "",
            "## Conversion",
            "",
            f"- source_safetensors_on_disk_gib: `{conv['source_safetensors_on_disk_gib']}`",
            f"- source_bf16_linear_gib_est: `{conv['source_bf16_linear_gib_est']}`",
            f"- resident_quantized_linear_gib: `{conv['resident_quantized_linear_gib']}`",
        ])
    drift = receipt.get("p1_drift")
    if drift:
        lines.extend([
            "",
            "## P1 Drift",
            "",
            f"- max_abs_delta_logit: `{drift['max_abs_delta_logit']}`",
            f"- top5_exact: `{drift['top5_exact_steps']}/{drift['top5_total_steps']}`",
            f"- multi_logit_drift_stop: `{drift['multi_logit_drift_stop']}`",
        ])
    if sanity:
        lines.extend([
            "",
            "## Sanity",
            "",
            f"- prompt: `{sanity['prompt_text']}`",
            f"- ids: `{sanity['generated_ids']}`",
            f"- text: {sanity['generated_text']!r}",
            f"- value_recovered: `{sanity['value_recovered']}`",
            f"- class: `{sanity['class']}`",
        ])
    if table:
        lines.extend([
            "",
            "## Grid",
            "",
            "| arm | width | live_shift | mount | cur_mount_n | value | class | ids | text |",
            "|---|---:|---:|---|---:|---|---|---|---|",
        ])
        for row in table:
            text = (row.get("text") or "").replace("|", "\\|").replace("\n", "\\n")
            if len(text) > 90:
                text = text[:87] + "..."
            lines.append(
                f"| {row['label']} | {row.get('width')} | {row.get('live_shift')} | "
                f"{row.get('mount')} | {row.get('cur_mount_n')} | {row['value_recovered']} | "
                f"{row['class']} | `{row['ids']}` | {text!r} |"
            )
    lines.extend([
        "",
        f"Artifacts: `{out_dir}`",
        "Script: `scripts/trinity_t1_natural_absolute.py`",
    ])
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--ngen", type=int, default=NGEN)
    p.add_argument("--wall-rail-s", type=float, default=GPU_WALL_RAIL_S)
    p.add_argument("--weight-mode", choices=("int4", "int8"), default="int8")
    p.add_argument("--reference-dir", type=Path, default=REF_DIR)
    p.add_argument("--drift-steps", type=int, default=DRIFT_STEPS)
    p.add_argument("--drift-stop-max-abs", type=float, default=DRIFT_STOP_MAX_ABS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or OUT_ROOT / f"natural_absolute_{stamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()
    vram = VramSampler()
    vram.start()
    vram.sample("start")

    print(
        f"[T1N] load {args.weight_mode.upper()} Trinity Nano from {args.model_dir}",
        flush=True,
    )
    model, model_info, load_s, dtype_meta = W.load_model(
        args.model_dir, weight_mode=args.weight_mode,
    )
    vram.sample("after_load")
    conv = conversion_receipt(args.model_dir, model_info)
    write_json(out_dir / "conversion_receipt.json", conv)
    print(f"[T1N] load_s={load_s:.1f} dtype={dtype_meta['mode_label']}", flush=True)
    if load_s > args.wall_rail_s:
        vram_summary = vram.stop()
        payload = {
            "verdict": {
                "verdict": "ABORTED_LOAD_OVER_RAIL",
                "reason": "model load exceeded single-run wall rail",
                "load_s": load_s,
            },
            "wall_rail_s": args.wall_rail_s,
            "conversion_receipt": conv,
            "vram_samples": vram_summary,
        }
        write_json(out_dir / "FINAL_VERDICT.json", payload)
        return 2

    tok = W.AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True,
    )
    adapter = W.TrinityArenaModelAdapter(model)
    adapter.configure_moe_empty_cache(0)

    run_config = {
        "experiment": "T1_absolute_natural_continuation",
        "chat_template_used": False,
        "model_dir": str(args.model_dir),
        "weight_mode": args.weight_mode,
        "compute": dtype_meta,
        "conversion_receipt": conv,
        "p1_drift_path": str(out_dir / "p1_bf16_vs_int8_drift.json"),
        "apa": "OFF_standard_everywhere",
        "plant_text": PLANT_TEXT,
        "filler_text": FILLER_TEXT,
        "readout_prompt": READOUT_PROMPT,
        "sink_text": SINK_TEXT,
        "widths": list(WIDTHS),
        "ngen": int(args.ngen),
        "value": FACT_VALUE,
        "model_info": model_info,
        "adapter_hooks": adapter._hook_meta,
        "attn_mode": adapter._attn_mode,
        "rails": {
            "single_gpu_run_wall_s": args.wall_rail_s,
            "project_tensor_product_diff": "core/trinity_nano_tc.py INT8 weight-mode option",
            "drift_stop_max_abs_delta_logit": args.drift_stop_max_abs,
        },
        "load_s": round(load_s, 3),
    }
    write_json(out_dir / "run_config.json", run_config)

    print("[T1N] P1 BF16-vs-INT8 teacher-forced drift gate", flush=True)
    p1_drift = run_p1_drift(
        adapter,
        args.reference_dir,
        steps=int(args.drift_steps),
        stop_max_abs=float(args.drift_stop_max_abs),
    )
    vram.sample("after_p1_drift")
    write_json(out_dir / "p1_bf16_vs_int8_drift.json", p1_drift)
    print(
        f"[T1N]   P1 max|dlogit|={p1_drift['max_abs_delta_logit']} "
        f"top5={p1_drift['top5_exact_steps']}/{p1_drift['top5_total_steps']} "
        f"stop={p1_drift['multi_logit_drift_stop']}",
        flush=True,
    )
    if p1_drift["multi_logit_drift_stop"]:
        verdict = {
            "verdict": "INT8_DRIFT_STOP",
            "reason": (
                "INT8-resident P1 teacher-forced logits exceeded the "
                f"registered multi-logit drift rail "
                f"({p1_drift['max_abs_delta_logit']} > "
                f"{p1_drift['stop_max_abs_delta_logit']}); stop before sanity/grid. "
                "This points at the quant/dequant path itself, not INT4 bit width."
            ),
            "sanity_passed": False,
            "grid_ran": False,
        }
        vram_summary = vram.stop()
        receipt = {
            "verdict": verdict,
            "conversion_receipt": conv,
            "p1_drift": p1_drift,
            "run_config_path": str(out_dir / "run_config.json"),
            "total_wall_s": round(time.perf_counter() - run_t0, 3),
            "vram_samples": vram_summary,
            "honest_residuals": [
                "T1 sanity/grid not run because the registered INT8 drift rail fired.",
                "The BF16 side of the drift receipt is the saved HF P1 reference artifact.",
                "VRAM peak is sampled nvidia-smi, not a TensorCUDA allocator peak.",
            ],
        }
        write_json(out_dir / "receipt.json", receipt)
        write_json(out_dir / "FINAL_VERDICT.json", verdict)
        write_summary(out_dir, receipt)
        del adapter, model
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
        return 3

    sanity_prompt = PLANT_TEXT + "\n" + READOUT_PROMPT
    print("[T1N] sanity no-arena live-context readout", flush=True)
    sanity = direct_greedy(
        adapter,
        tok,
        sanity_prompt,
        int(args.ngen),
        "sanity_live_context_no_arena",
        weight_mode=args.weight_mode,
    )
    vram.sample("after_sanity")
    write_json(out_dir / "sanity_live_context_no_arena.json", sanity)
    print(
        f"[T1N]   sanity value={sanity['value_recovered']} "
        f"class={sanity['class']} text={sanity['generated_text']!r}",
        flush=True,
    )

    cases: list[dict[str, Any]] = []
    pure_baseline = None
    if sanity["value_recovered"]:
        print("[T1N] pure baseline no fact/no arena", flush=True)
        pure_baseline = direct_greedy(
            adapter,
            tok,
            READOUT_PROMPT,
            int(args.ngen),
            "pure_baseline_no_context",
            weight_mode=args.weight_mode,
        )
        vram.sample("after_pure_baseline")
        write_json(out_dir / "case_pure_baseline_no_context.json", pure_baseline)
        print(
            f"[T1N]   pure value={pure_baseline['value_recovered']} "
            f"class={pure_baseline['class']} text={pure_baseline['generated_text']!r}",
            flush=True,
        )
        for width in WIDTHS:
            for mount in (True, False):
                elapsed = time.perf_counter() - run_t0
                if elapsed > args.wall_rail_s:
                    raise RuntimeError(
                        f"overall run rail exceeded before w{width}_{mount}: "
                        f"{elapsed:.3f}s"
                    )
                print(f"[T1N] case width={width} mount={mount}", flush=True)
                row = run_grid_case(
                    adapter, tok, width=int(width), mount=bool(mount),
                    ngen=int(args.ngen),
                )
                cases.append(row)
                write_json(out_dir / f"case_{row['label']}.json", row)
                gen = row["generation"]
                print(
                    f"[T1N]   shift={row['live_shift']} value={gen['value_recovered']} "
                    f"class={gen['class']} text={gen['generated_text']!r} "
                    f"wall={row['wall_s']}s",
                    flush=True,
                )
                if row["wall_s"] > args.wall_rail_s:
                    raise RuntimeError(
                        f"single case wall rail exceeded: {row['label']} "
                        f"{row['wall_s']}s"
                    )
                vram.sample(f"after_{row['label']}")

    verdict = decide(sanity, cases)
    grid_table = []
    if pure_baseline is not None:
        grid_table.append({
            "label": pure_baseline["label"],
            "width": None,
            "live_shift": None,
            "mount": False,
            "cur_mount_n": 0,
            "fact_evicted": None,
            "picks": [],
            "picks_forced": False,
            "ids": pure_baseline["generated_ids"],
            "text": pure_baseline["generated_text"],
            "raw": pure_baseline["generated_text_raw"],
            "value_recovered": pure_baseline["value_recovered"],
            "class": pure_baseline["class"],
            "wall_s": pure_baseline["wall_s"],
        })
    for row in cases:
        gen = row["generation"]
        grid_table.append({
            "label": row["label"],
            "width": row["width"],
            "live_shift": row["live_shift"],
            "mount": row["mount"],
            "cur_mount_n": row["cur_mount_n"],
            "fact_evicted": row["fact_evicted"],
            "picks": row["picks"],
            "picks_forced": row["picks_forced"],
            "ids": gen["generated_ids"],
            "text": gen["generated_text"],
            "raw": gen["generated_text_raw"],
            "value_recovered": gen["value_recovered"],
            "class": gen["class"],
            "wall_s": row["wall_s"],
        })

    receipt = {
        "verdict": verdict,
        "conversion_receipt": conv,
        "p1_drift": p1_drift,
        "sanity": sanity,
        "pure_baseline": pure_baseline,
        "grid_table": grid_table,
        "cases": cases,
        "run_config_path": str(out_dir / "run_config.json"),
        "total_wall_s": round(time.perf_counter() - run_t0, 3),
        "vram_samples": vram.stop(),
        "honest_residuals": [
            "This receipt uses INT8 weights + fp32 compute after the P1 BF16 reference drift gate.",
            "The BF16 side of the drift receipt is the saved HF P1 reference artifact, not a same-process bf16 stream rerun.",
            "The adapter hook is in-memory harness code; Project-Tensor product edit is limited to the INT8 weight-mode loader.",
            "Value recovery is exact code-shape normalized over generated continuation only.",
            "Controls prove graft-carrying only if their continuations miss the value.",
            "VRAM peak is sampled nvidia-smi, not a TensorCUDA allocator peak.",
        ],
    }
    write_json(out_dir / "receipt.json", receipt)
    write_json(out_dir / "FINAL_VERDICT.json", verdict)
    write_summary(out_dir, receipt)

    print(f"[T1N] VERDICT={verdict['verdict']}", flush=True)
    print(f"[T1N] {verdict['reason']}", flush=True)
    print(f"[T1N] artifacts -> {out_dir}", flush=True)

    del adapter, model
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
