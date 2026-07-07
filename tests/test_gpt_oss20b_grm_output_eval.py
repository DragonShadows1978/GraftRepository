import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gpt_oss20b_grm_output_eval import (  # noqa: E402
    canonical_value,
    evaluate_gate_artifact,
    find_value_rank,
    summarize_pair,
)


def test_canonical_value_handles_case_space_and_punctuation():
    assert canonical_value(" BLUE.") == "blue"
    assert canonical_value("\u2581Blue!") == "blue"
    assert canonical_value(None) == ""


def test_find_value_rank_matches_case_normalized_topk():
    top_tokens = [
        {"text": "blue", "logit": 24.0, "token_id": 1},
        {"text": "BLUE", "logit": 23.0, "token_id": 2},
    ]

    rank = find_value_rank(top_tokens, "BLUE")

    assert rank["rank"] == 0
    assert rank["text"] == "blue"
    assert rank["logit"] == 24.0


def test_summarize_pair_separates_value_hit_from_control_confound():
    control = {"top_token": {"text": "blue"}}
    mount = {
        "top_token": {"text": "blue"},
        "top_tokens": [{"text": "blue", "logit": 24.0}],
    }

    summary = summarize_pair(control=control, mount=mount, answer="BLUE")

    assert summary["normalized_top_hit"] is True
    assert summary["unconfounded_normalized_top_hit"] is False
    assert summary["normalized_answer_rank"] == 0


def test_summarize_pair_marks_stale_suppression():
    control = {"top_token": {"text": "GPT"}}
    mount = {
        "top_token": {"text": "blue"},
        "top_tokens": [
            {"text": "blue", "logit": 23.0},
            {"text": "RED", "logit": 15.0},
        ],
    }

    summary = summarize_pair(
        control=control,
        mount=mount,
        answer="BLUE",
        stale_answers=["RED"],
    )

    assert summary["normalized_top_hit"] is True
    assert summary["unconfounded_normalized_top_hit"] is True
    assert summary["stale_top_hit"] is False
    assert summary["stale_ranks"][0]["rank"] == 1


def test_evaluate_gate_artifact_counts_normalized_hits(tmp_path):
    artifact = {
        "schema": "gpt_oss_20b_preference_graft_gate_v1",
        "classification": "fail",
        "hit_count": 1,
        "runs": {
            "items": [
                {
                    "id": "preferred_color",
                    "label": "user preferred color",
                    "answer": "BLUE",
                    "control": {
                        "summary": {
                            "top_token": {"text": "blue"},
                            "top_tokens": [{"text": "blue"}],
                        }
                    },
                    "mount": {
                        "summary": {
                            "top_token": {"text": "blue"},
                            "top_tokens": [{"text": "blue"}],
                        }
                    },
                },
                {
                    "id": "preferred_signal",
                    "label": "user preferred signal",
                    "answer": "EMBER",
                    "control": {
                        "summary": {
                            "top_token": {"text": "B"},
                            "top_tokens": [{"text": "B"}],
                        }
                    },
                    "mount": {
                        "summary": {
                            "top_token": {"text": "EMBER"},
                            "top_tokens": [{"text": "EMBER"}],
                        }
                    },
                },
            ]
        },
    }
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    audit = evaluate_gate_artifact(path)

    assert audit["probe_count"] == 2
    assert audit["normalized_value_top1_hits"] == 2
    assert audit["normalized_unconfounded_hits"] == 1
    assert audit["classification"]["normalized_value_top1"] == "pass"
    assert audit["classification"]["normalized_unconfounded_top1"] == "fail"
