#!/usr/bin/env python3
"""Normalize and audit GPT-OSS GRM recall artifacts.

The H6 graft gates intentionally record strict first-token exact matches. This
helper keeps that raw evidence intact, then adds a separate output-policy view:
case-normalized top-1 hits, unconfounded normalized hits, stale-value checks,
and normalized top-k value presence.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STRIP_CHARS = " \t\r\n\"'`.,:;!?()[]{}"


def canonical_value(text: Any) -> str:
    """Return a conservative canonical form for single-value recall checks."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u2581", " ")
    value = re.sub(r"\s+", " ", value).strip(STRIP_CHARS)
    return value.casefold()


def answer_forms(answer: str, aliases: list[str] | None = None) -> set[str]:
    forms = {canonical_value(answer)}
    for alias in aliases or []:
        form = canonical_value(alias)
        if form:
            forms.add(form)
    return {item for item in forms if item}


def find_value_rank(
    top_tokens: list[dict[str, Any]],
    answer: str,
    *,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    forms = answer_forms(answer, aliases)
    for idx, row in enumerate(top_tokens):
        if canonical_value(row.get("text")) in forms:
            return {
                "rank": int(idx),
                "text": row.get("text"),
                "logit": row.get("logit"),
                "token_id": row.get("token_id"),
            }
    return {"rank": None, "text": None, "logit": None, "token_id": None}


def summarize_pair(
    *,
    control: dict[str, Any],
    mount: dict[str, Any],
    answer: str,
    stale_answers: list[str] | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    control_top = (control.get("top_token") or {}).get("text")
    mount_top = (mount.get("top_token") or {}).get("text")
    top_tokens = mount.get("top_tokens") or []

    answer_canon = canonical_value(answer)
    control_canon = canonical_value(control_top)
    mount_canon = canonical_value(mount_top)
    forms = answer_forms(answer, aliases)
    rank = find_value_rank(top_tokens, answer, aliases=aliases)

    stale_rows = []
    stale_top_hit = False
    for stale in stale_answers or []:
        stale_rank = find_value_rank(top_tokens, stale)
        stale_row = {
            "answer": stale,
            "normalized_answer": canonical_value(stale),
            "rank": stale_rank["rank"],
            "text": stale_rank["text"],
            "logit": stale_rank["logit"],
        }
        stale_rows.append(stale_row)
        if mount_canon and mount_canon == canonical_value(stale):
            stale_top_hit = True

    exact_top_hit = mount_top == answer
    normalized_top_hit = bool(mount_canon and mount_canon in forms)
    return {
        "answer": answer,
        "normalized_answer": answer_canon,
        "control_top": control_top,
        "mount_top": mount_top,
        "normalized_control_top": control_canon,
        "normalized_mount_top": mount_canon,
        "exact_top_hit": bool(exact_top_hit),
        "normalized_top_hit": bool(normalized_top_hit),
        "unconfounded_exact_top_hit": bool(exact_top_hit and control_top != answer),
        "unconfounded_normalized_top_hit": bool(
            normalized_top_hit and control_canon not in forms
        ),
        "normalized_answer_rank": rank["rank"],
        "normalized_answer_text": rank["text"],
        "normalized_answer_logit": rank["logit"],
        "normalized_answer_in_topk": rank["rank"] is not None,
        "stale_top_hit": bool(stale_top_hit),
        "stale_ranks": stale_rows,
    }


def iter_gate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    schema = str(payload.get("schema") or "")
    runs = payload.get("runs")
    if schema == "gpt_oss_20b_multifact_addressing_gate_v1":
        return list(runs or [])
    if not isinstance(runs, dict):
        return []
    if schema in {
        "gpt_oss_20b_preference_graft_gate_v1",
        "gpt_oss_20b_supersession_graft_gate_v1",
    }:
        return list(runs.get("items") or [])
    if schema == "gpt_oss_20b_multifact_graft_gate_v1":
        return list(runs.get("facts") or [])
    return []


def evaluate_gate_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = iter_gate_rows(payload)
    evaluated = []
    for row in rows:
        control = ((row.get("control") or {}).get("summary") or {})
        mount = ((row.get("mount") or {}).get("summary") or {})
        answer = str(row.get("answer") or "")
        stale = []
        if row.get("old_answer") is not None:
            stale.append(str(row["old_answer"]))
        item = {
            "variant": row.get("variant"),
            "id": row.get("id"),
            "label": row.get("label"),
            "old_answer": row.get("old_answer"),
            **summarize_pair(
                control=control,
                mount=mount,
                answer=answer,
                stale_answers=stale,
            ),
        }
        evaluated.append(item)

    total = len(evaluated)
    exact_hits = sum(1 for row in evaluated if row["unconfounded_exact_top_hit"])
    normalized_hits = sum(
        1 for row in evaluated if row["unconfounded_normalized_top_hit"]
    )
    normalized_value_hits = sum(1 for row in evaluated if row["normalized_top_hit"])
    topk_value_hits = sum(1 for row in evaluated if row["normalized_answer_in_topk"])
    stale_top_hits = sum(1 for row in evaluated if row["stale_top_hit"])
    return {
        "artifact": str(path),
        "schema": payload.get("schema"),
        "source_classification": payload.get("classification"),
        "source_hit_count": payload.get("hit_count"),
        "probe_count": total,
        "exact_unconfounded_hits": exact_hits,
        "normalized_unconfounded_hits": normalized_hits,
        "normalized_value_top1_hits": normalized_value_hits,
        "normalized_value_topk_hits": topk_value_hits,
        "stale_top_hits": stale_top_hits,
        "classification": {
            "exact_unconfounded_top1": "pass" if total and exact_hits == total else "fail",
            "normalized_unconfounded_top1": (
                "pass" if total and normalized_hits == total else "fail"
            ),
            "normalized_value_top1": (
                "pass" if total and normalized_value_hits == total else "fail"
            ),
            "normalized_value_topk": (
                "pass" if total and topk_value_hits == total else "fail"
            ),
            "stale_suppression": "pass" if stale_top_hits == 0 else "fail",
        },
        "items": evaluated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GPT-OSS GRM output artifacts with normalized value checks."
    )
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "artifacts" / "gpt_oss_20b" / f"h6_output_eval_{stamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.output.expanduser().resolve() if args.output else default_output()
    audits = [evaluate_gate_artifact(path.expanduser().resolve()) for path in args.artifacts]
    payload = {
        "schema": "gpt_oss_20b_grm_output_eval_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": [str(path.expanduser().resolve()) for path in args.artifacts],
        "audits": audits,
    }
    write_json(out, payload)
    print(f"artifact={out}", flush=True)
    print(
        json.dumps(
            [
                {
                    "artifact": Path(row["artifact"]).name,
                    "schema": row["schema"],
                    "source_classification": row["source_classification"],
                    "exact_unconfounded_hits": row["exact_unconfounded_hits"],
                    "normalized_unconfounded_hits": row["normalized_unconfounded_hits"],
                    "normalized_value_top1_hits": row["normalized_value_top1_hits"],
                    "normalized_value_topk_hits": row["normalized_value_topk_hits"],
                    "probe_count": row["probe_count"],
                    "classification": row["classification"],
                }
                for row in audits
            ]
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
