import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gpt_oss20b_grm_output_eval import (  # noqa: E402
    canonical_value,
    contains_value_text,
    evaluate_gate_artifact,
    find_value_rank,
    generated_suffix_from_payload,
    summarize_pair,
)


def test_canonical_value_handles_case_space_and_punctuation():
    assert canonical_value(" BLUE.") == "blue"
    assert canonical_value("\u2581Blue!") == "blue"
    assert canonical_value("7391\u20112048") == "7391-2048"
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


def test_contains_value_text_uses_word_boundaries():
    assert contains_value_text("The stored value is BLUE.", "BLUE") is True
    assert contains_value_text("The stored value is BLUEPRINT.", "BLUE") is False
    assert contains_value_text("The stored value is blue.", "BLUE") is True
    assert contains_value_text("The stored value is 7391\u20112048.", "7391-2048") is True


def test_generated_suffix_from_payload_removes_prompt_prefix():
    payload = {"initial_prompt": "Question:", "final_text": "Question: BLUE"}

    assert generated_suffix_from_payload(payload) == " BLUE"


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


def test_summarize_pair_scores_generated_text(tmp_path):
    control_artifact = tmp_path / "control.json"
    mount_artifact = tmp_path / "mount.json"
    control_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: I do not know."}),
        encoding="utf-8",
    )
    mount_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: The stored value is BLUE."}),
        encoding="utf-8",
    )
    control = {"top_token": {"text": "I"}, "artifact": str(control_artifact)}
    mount = {
        "top_token": {"text": "The"},
        "top_tokens": [{"text": "The"}],
        "artifact": str(mount_artifact),
    }

    summary = summarize_pair(control=control, mount=mount, answer="BLUE")

    assert summary["normalized_top_hit"] is False
    assert summary["mount_generated_contains_value"] is True
    assert summary["unconfounded_generated_value_hit"] is True


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
    assert audit["generated_value_hits"] == 0
    assert audit["classification"]["normalized_value_top1"] == "pass"
    assert audit["classification"]["normalized_unconfounded_top1"] == "fail"


def test_evaluate_exact_value_artifact_counts_generated_hits(tmp_path):
    control_artifact = tmp_path / "control.json"
    mount_artifact = tmp_path / "mount.json"
    control_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: I do not know."}),
        encoding="utf-8",
    )
    mount_artifact.write_text(
        json.dumps(
            {"initial_prompt": "Q:", "final_text": "Q: The exact code is ZX-47B."}
        ),
        encoding="utf-8",
    )
    artifact = {
        "schema": "gpt_oss_20b_exact_value_graft_gate_v1",
        "classification": "fail",
        "hit_count": 0,
        "runs": {
            "items": [
                {
                    "id": "asset_code",
                    "label": "GPT-OSS asset code",
                    "answer": "ZX-47B",
                    "control": {
                        "summary": {
                            "top_token": {"text": "I"},
                            "artifact": str(control_artifact),
                        }
                    },
                    "mount": {
                        "summary": {
                            "top_token": {"text": "The"},
                            "top_tokens": [{"text": "The"}],
                            "artifact": str(mount_artifact),
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    audit = evaluate_gate_artifact(path)

    assert audit["probe_count"] == 1
    assert audit["exact_unconfounded_hits"] == 0
    assert audit["generated_unconfounded_hits"] == 1
    assert audit["classification"]["generated_unconfounded_value"] == "pass"


def test_evaluate_instruction_retention_artifact_counts_generated_hits(tmp_path):
    control_artifact = tmp_path / "control.json"
    mount_artifact = tmp_path / "mount.json"
    control_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: I do not know."}),
        encoding="utf-8",
    )
    mount_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: LUMEN-42"}),
        encoding="utf-8",
    )
    artifact = {
        "schema": "gpt_oss_20b_instruction_retention_gate_v1",
        "classification": "fail",
        "hit_count": 0,
        "runs": {
            "items": [
                {
                    "id": "retention_instruction_a",
                    "label": "retained instruction A response string",
                    "answer": "LUMEN-42",
                    "control": {
                        "summary": {
                            "top_token": {"text": "I"},
                            "artifact": str(control_artifact),
                        }
                    },
                    "mount": {
                        "summary": {
                            "top_token": {"text": "L"},
                            "top_tokens": [{"text": "L"}],
                            "artifact": str(mount_artifact),
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "instruction.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    audit = evaluate_gate_artifact(path)

    assert audit["probe_count"] == 1
    assert audit["generated_unconfounded_hits"] == 1
    assert audit["classification"]["generated_unconfounded_value"] == "pass"


def test_evaluate_repetition_drift_artifact_counts_generated_hits(tmp_path):
    control_artifact = tmp_path / "control.json"
    mount_artifact = tmp_path / "mount.json"
    control_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: I do not know."}),
        encoding="utf-8",
    )
    mount_artifact.write_text(
        json.dumps({"initial_prompt": "Q:", "final_text": "Q: ORBIT-7"}),
        encoding="utf-8",
    )
    artifact = {
        "schema": "gpt_oss_20b_repetition_drift_gate_v1",
        "classification": "fail",
        "hit_count": 0,
        "runs": {
            "items": [
                {
                    "id": "retention_instruction_b",
                    "repeat_index": 0,
                    "label": "retained instruction B response string",
                    "answer": "ORBIT-7",
                    "control": {
                        "summary": {
                            "top_token": {"text": "I"},
                            "artifact": str(control_artifact),
                        }
                    },
                    "mount": {
                        "summary": {
                            "top_token": {"text": "OR"},
                            "top_tokens": [{"text": "OR"}],
                            "artifact": str(mount_artifact),
                        }
                    },
                }
            ]
        },
    }
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    audit = evaluate_gate_artifact(path)

    assert audit["probe_count"] == 1
    assert audit["generated_unconfounded_hits"] == 1
    assert audit["classification"]["generated_unconfounded_value"] == "pass"
