#!/usr/bin/env python3
"""GPT-OSS bulk graft capture/remount gate.

This is an H5 helper, not an H5 result by itself. It builds a real-token prompt
with a needle near the end, captures GPT-OSS pre-RoPE K/V shards, then asks a
short live query with and without the mounted graft and scores gold/decoy first
tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-OSS bulk graft gate.")
    parser.add_argument("--model-dir", default=SNAPSHOT)
    parser.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--target-tokens", type=int, default=131072)
    parser.add_argument(
        "--needle-text",
        default=(
            "\nCLASSIFIED NEEDLE. The GPT-OSS vault keyword is BLUE.\n"
            "END NEEDLE.\n"
        ),
    )
    parser.add_argument(
        "--query",
        default="User: What is the GPT-OSS vault keyword?\nAssistant: The keyword is",
    )
    parser.add_argument("--gold-candidate", default=" BLUE")
    parser.add_argument(
        "--decoy-candidate",
        action="append",
        default=[" RED", " GREEN", " BLACK", " WHITE"],
    )
    parser.add_argument("--graft-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--attention-mode",
        choices=("standard", "apa_selective"),
        default="apa_selective",
    )
    parser.add_argument("--apa-layer-scope", choices=("full", "all"), default="full")
    parser.add_argument("--refine-percentile", type=float, default=0.15)
    parser.add_argument("--bulk-bits", type=int, default=8)
    parser.add_argument(
        "--expert-mode",
        choices=("resident_packed_mxfp4", "packed_mxfp4", "dequant"),
        default="resident_packed_mxfp4",
    )
    parser.add_argument("--route-detail", choices=("full", "summary"), default="summary")
    parser.add_argument("--expert-empty-cache-interval", type=int, default=0)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build prompts and receipts without launching streamed GPU runs",
    )
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h5_bulk_graft_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_corpus(root: Path) -> tuple[str, dict[str, Any]]:
    root = root.expanduser().resolve()
    files = sorted(root.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no markdown corpus files found in {root}")
    parts: list[str] = []
    h = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        rel = str(path.relative_to(root))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(data)
        h.update(b"\0")
        parts.append(data.decode("utf-8", errors="ignore"))
    return "\n\n".join(parts), {
        "kind": "directory",
        "path": str(root),
        "file_count": len(files),
        "sha256": h.hexdigest(),
    }


def token_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def build_capture_prompt(
    *,
    tokenizer,
    corpus_text: str,
    needle_text: str,
    target_tokens: int,
) -> tuple[str, list[int], dict[str, Any]]:
    corpus_ids = token_ids(tokenizer, corpus_text)
    needle_ids = token_ids(tokenizer, needle_text)
    fill_tokens = int(target_tokens) - len(needle_ids)
    if fill_tokens < 1:
        raise ValueError(
            f"target_tokens={target_tokens} is smaller than needle token count "
            f"{len(needle_ids)}"
        )
    if len(corpus_ids) < fill_tokens:
        raise ValueError(
            f"corpus has {len(corpus_ids)} tokens but needs {fill_tokens}; "
            "refusing to repeat tokens"
        )
    ids = corpus_ids[:fill_tokens] + needle_ids
    text = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text, ids, {
        "target_tokens": int(target_tokens),
        "actual_tokens": len(ids),
        "corpus_fill_tokens": int(fill_tokens),
        "needle_tokens": len(needle_ids),
        "needle_offset_tokens": int(fill_tokens),
        "needle_text_sha256": sha256_text(needle_text),
        "input_mode": "direct_token_ids",
    }


def run_child(cmd: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "wall_seconds": time.perf_counter() - started,
    }


def stream_cmd(
    *,
    model_dir: Path,
    prompt_file: Path,
    max_tokens: int,
    output: Path,
    args: argparse.Namespace,
    graft_dir: Path | None = None,
    capture_dir: Path | None = None,
    input_ids_file: Path | None = None,
    candidates: list[str] | None = None,
    skip_lm_head: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "gpt_oss20b_stream_forward_smoke.py"),
        "--model-dir",
        str(model_dir),
        "--prompt-file",
        str(prompt_file),
        "--max-tokens",
        str(max_tokens),
        "--attention-mode",
        args.attention_mode,
        "--apa-layer-scope",
        args.apa_layer_scope,
        "--refine-percentile",
        str(args.refine_percentile),
        "--bulk-bits",
        str(args.bulk_bits),
        "--expert-mode",
        args.expert_mode,
        "--route-detail",
        args.route_detail,
        "--expert-empty-cache-interval",
        str(args.expert_empty_cache_interval),
        "--output",
        str(output),
    ]
    if args.max_layers is not None:
        cmd.extend(["--max-layers", str(args.max_layers)])
    if input_ids_file is not None:
        cmd.extend(["--input-ids-file", str(input_ids_file)])
    if skip_lm_head:
        cmd.append("--skip-lm-head")
    if capture_dir is not None:
        cmd.extend(["--capture-graft-dir", str(capture_dir)])
    if graft_dir is not None:
        cmd.extend(["--mount-graft-dir", str(graft_dir)])
    for cand in candidates or []:
        cmd.extend(["--candidate-text", cand])
    return cmd


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_map(artifact: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not artifact:
        return {}
    return {
        str(item.get("candidate_text")): item
        for item in artifact.get("candidate_scores", [])
        if "candidate_text" in item
    }


def summarize_scores(
    artifact: dict[str, Any] | None,
    *,
    gold: str,
    decoys: list[str],
) -> dict[str, Any]:
    scores = candidate_map(artifact)
    gold_item = scores.get(gold)
    decoy_items = [scores.get(x) for x in decoys if scores.get(x) is not None]
    best_decoy = None
    if decoy_items:
        best_decoy = max(
            decoy_items,
            key=lambda item: float(item.get("first_token_logit", float("-inf"))),
        )
    gold_logit = (
        None if gold_item is None else float(gold_item.get("first_token_logit"))
    )
    best_decoy_logit = (
        None if best_decoy is None else float(best_decoy.get("first_token_logit"))
    )
    return {
        "gold": gold_item,
        "best_decoy": best_decoy,
        "gold_minus_best_decoy_logit": (
            None
            if gold_logit is None or best_decoy_logit is None
            else gold_logit - best_decoy_logit
        ),
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out = args.output.expanduser().resolve() if args.output else default_output()
    run_dir = out.parent / out.stem
    graft_dir = (
        args.graft_dir.expanduser().resolve()
        if args.graft_dir is not None
        else run_dir / "graft"
    )
    capture_prompt_file = run_dir / "capture_prompt.txt"
    capture_ids_file = run_dir / "capture_ids.json"
    query_file = run_dir / "query_prompt.txt"
    capture_artifact = run_dir / "capture_forward.json"
    control_artifact = run_dir / "control_forward.json"
    mount_artifact = run_dir / "mount_forward.json"
    candidates = [args.gold_candidate] + list(args.decoy_candidate)

    payload: dict[str, Any] = {
        "schema": "gpt_oss_20b_bulk_graft_gate_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "target_tokens": int(args.target_tokens),
        "graft_dir": str(graft_dir),
        "capture_prompt_file": str(capture_prompt_file),
        "capture_ids_file": str(capture_ids_file),
        "query_file": str(query_file),
        "gold_candidate": args.gold_candidate,
        "decoy_candidates": list(args.decoy_candidate),
        "status": "starting",
        "runs": {},
        "note": (
            "gold-vs-decoy candidate scoring with and without mounted pre-RoPE "
            "K/V grafts; a pass here is H5 logit-access evidence, not greedy "
            "open-generation recall"
        ),
    }
    write_json(out, payload)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    corpus_text, corpus_info = read_corpus(args.corpus_dir)
    capture_text, capture_ids, prompt_info = build_capture_prompt(
        tokenizer=tokenizer,
        corpus_text=corpus_text,
        needle_text=args.needle_text,
        target_tokens=args.target_tokens,
    )
    capture_prompt_file.parent.mkdir(parents=True, exist_ok=True)
    capture_prompt_file.write_text(capture_text, encoding="utf-8")
    capture_ids_file.write_text(json.dumps(capture_ids), encoding="utf-8")
    query_file.write_text(args.query, encoding="utf-8")
    query_tokens = len(token_ids(tokenizer, args.query))

    payload.update(
        {
            "status": "running",
            "corpus": corpus_info,
            "capture_prompt": prompt_info,
            "query_tokens": int(query_tokens),
            "candidate_tokenization": {
                cand: token_ids(tokenizer, cand) for cand in candidates
            },
        }
    )
    write_json(out, payload)

    if args.dry_run:
        payload["status"] = "dry_run"
        payload["classification"] = "not_run"
        write_json(out, payload)
        print(f"artifact={out}", flush=True)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "capture_tokens": prompt_info["actual_tokens"],
                    "query_tokens": query_tokens,
                    "candidate_tokenization": payload["candidate_tokenization"],
                }
            ),
            flush=True,
        )
        return 0

    if not args.skip_capture:
        capture_cmd = stream_cmd(
            model_dir=model_dir,
            prompt_file=capture_prompt_file,
            max_tokens=args.target_tokens,
            output=capture_artifact,
            args=args,
            capture_dir=graft_dir,
            input_ids_file=capture_ids_file,
            skip_lm_head=True,
        )
        payload["runs"]["capture"] = run_child(capture_cmd)
        write_json(out, payload)
        if payload["runs"]["capture"]["returncode"] != 0:
            payload["status"] = "capture_error"
            write_json(out, payload)
            print(f"artifact={out}", flush=True)
            return int(payload["runs"]["capture"]["returncode"]) or 1

    if not args.skip_control:
        control_cmd = stream_cmd(
            model_dir=model_dir,
            prompt_file=query_file,
            max_tokens=query_tokens,
            output=control_artifact,
            args=args,
            candidates=candidates,
        )
        payload["runs"]["control"] = run_child(control_cmd)
        write_json(out, payload)

    mount_cmd = stream_cmd(
        model_dir=model_dir,
        prompt_file=query_file,
        max_tokens=query_tokens,
        output=mount_artifact,
        args=args,
        graft_dir=graft_dir,
        candidates=candidates,
    )
    payload["runs"]["mount"] = run_child(mount_cmd)

    control_payload = load_artifact(control_artifact)
    mount_payload = load_artifact(mount_artifact)
    payload["artifacts"] = {
        "capture": str(capture_artifact),
        "control": str(control_artifact),
        "mount": str(mount_artifact),
        "graft_manifest": str(graft_dir / "manifest.json"),
    }
    payload["score_summary"] = {
        "control": summarize_scores(
            control_payload,
            gold=args.gold_candidate,
            decoys=list(args.decoy_candidate),
        ),
        "mount": summarize_scores(
            mount_payload,
            gold=args.gold_candidate,
            decoys=list(args.decoy_candidate),
        ),
    }
    mount_margin = payload["score_summary"]["mount"]["gold_minus_best_decoy_logit"]
    control_margin = payload["score_summary"]["control"][
        "gold_minus_best_decoy_logit"
    ]
    payload["classification"] = (
        "pass"
        if mount_margin is not None
        and mount_margin > 0.0
        and (control_margin is None or mount_margin > control_margin)
        else "fail"
    )
    payload["status"] = "ok"
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "classification": payload["classification"],
                "mount_margin": mount_margin,
                "control_margin": control_margin,
            }
        ),
        flush=True,
    )
    return 0 if payload["classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
