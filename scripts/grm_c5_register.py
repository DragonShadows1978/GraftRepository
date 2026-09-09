"""Freeze C5 inputs before gates; foreground text reads only.

Prior art: GRM DET/SC/WC immutable registration and SHA-256 receipts (2026).
Borrowed: frozen inputs, paired arms, independent expected labels. Ours:
receipt selection and constructed adversaries below. External lead: Pearl,
Causality (2000), controlled interventions; unverified — lead to check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/grm_c5"
ARCHIVE = Path("/mnt/ForgeRealm/GraftRepository")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def write_new(path, data):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "registration.json").exists():
        raise RuntimeError("Immutable registration exists; use a separate amendment")
    inputs = {}

    def read(path):
        path = Path(path)
        inputs[str(path)] = record(path)
        return json.loads(path.read_text())

    def lines(path):
        inputs[str(path)] = record(path)
        return [json.loads(line) for line in path.read_text().splitlines()]

    fixtures = []
    # Prior art: EB1/WC1's original final longhorizon scorecards (GRM, 2026).
    # Select complete final sessions by named width, not measured outcome.
    sessions = {
        "EB1": "grm_eb1/g5_run/lh-5/session",
        "WC1-64": "grm_wc1_astra/width_64/longhorizon/run/lh-5/session",
        "WC1-96": "grm_wc1_astra/width_96/longhorizon/run/lh-5/session",
        **{f"WC1-{w}": f"grm_wc1_opus/runs/w{w}/lh_session/lh-5/session"
           for w in (128, 192, 256)},
    }
    for campaign, relative in sessions.items():
        session = ARCHIVE / "artifacts" / relative
        score_path = session / "probe_scorecard.json"
        score = read(score_path)
        nodes = read(session / "repository/manifest.json")["nodes"]
        transcript = {r["turn"]: r for r in lines(session / "transcript.jsonl")}
        instrumentation = {r["turn"]: r for r in lines(session / "instrumentation.jsonl")}
        for row in score["probes"]:
            turn = row["turn"]
            instr = instrumentation[turn]
            ids = row["mounted_ids"]
            assert row["answer"] == transcript[turn]["assistant"]
            assert ids == instr["mounts_after"]
            fixtures.append({
                "id": f"{campaign}:t{turn:02}", "class": "original",
                "evidence": "served_text_with_persisted_mount_text",
                "campaign": campaign, "question": transcript[turn]["user"],
                "answer": row["answer"], "texts": [nodes[i]["text"] for i in ids],
                "mounted_ids": ids, "expected": [row["expected"]],
                "recorded_correct": row["pass"], "desired_correct": True,
                "recorded_trips": instr["info"].get("_route_observation", {}).get("trips", []),
                "abstain_reason": instr["info"].get("abstain_reason"),
                "infer_calls": instr.get("infer_calls"),
                "source": str(score_path), "turn": turn,
                "source_node_id": row["eviction_check"]["source_node_id"],
                "source_node_text": nodes[row["eviction_check"]["source_node_id"]]["text"],
                "excluded_live_ids": instr["info"].get("_route_observation", {}).get("excluded_live_ids", []),
            })

    pair_path = ARCHIVE / "artifacts/grm_sc1_2/sc1_2_pair_e2e_t30_atlas_tone_a3b37c16ac28e23d.json"
    pair = read(pair_path)
    result = pair["result"]
    repo_ref = result["index_reconstruction"]["source_manifest"]
    repo_path = ARCHIVE / repo_ref["path"]
    nodes = read(repo_path)["nodes"]
    assert digest(repo_path) == repo_ref["sha256"]
    snapshot_ref = result["frozen_snapshot"]
    snap_path = ARCHIVE / snapshot_ref["path"]
    snapshot = read(snap_path)
    assert digest(snap_path) == snapshot_ref["sha256"]
    arm = result["arms"]["served"]["arm0"]
    assert arm["attempt_mounted_ids"] == snapshot["state"]["arena.cur_mounts"]
    fixtures.append({
        "id": "SC1:t30", "class": "original", "campaign": "SC1",
        "evidence": "served_text_and_frozen_mount_ids_with_hash_bound_repository_text",
        "question": result["question"], "answer": arm["served_answer"],
        "texts": [nodes[i]["text"] for i in arm["attempt_mounted_ids"]],
        "mounted_ids": arm["attempt_mounted_ids"],
        "expected": result["expected_values"], "recorded_correct": True,
        "desired_correct": True, "source": str(pair_path),
    })
    trip = result["arms"]["planted_miss"]["arm1"]["demand_trip"]
    fixtures.append({**fixtures[-1], "id": "SC1:t30-trip",
                     "evidence": "generated_demand_trip_not_selected_as_served",
                     "answer": trip["demand_trip_answer"],
                     "mounted_ids": trip["demand_trip_mounted_ids"],
                     "texts": [nodes[i]["text"] for i in trip["demand_trip_mounted_ids"]],
                     "recorded_grounded": trip["demand_trip_grounded"]})

    # RT1 stores served answers and mount IDs, but its summary and linked
    # receipts do not contain exact mounted texts (including split children).
    # Do not substitute fixture parent prose for the actual mounted child.
    rt_path = ARCHIVE / "artifacts/grm_rt1/grm_rt1_g2_results.json"
    rt = read(rt_path)
    for arm_name in ("rule_off", "rule_on"):
        for ref in rt[arm_name]["receipts"]:
            path = ARCHIVE / ref["path"]
            receipt = read(path)
            assert digest(path) == ref["sha256"]
            for row in receipt["probes"]:
                fixtures.append({
                    "id": f"RT1:{arm_name}:{row['probe_id']}", "campaign": "RT1",
                    "class": "original", "evidence": "served_text_only",
                    "question": row["question"], "answer": row["served_answer"],
                    "texts": None, "mounted_ids": row["mounted_ids"],
                    "expected": row["expected_values"],
                    "recorded_correct": row["verdict"]["correct"], "desired_correct": True,
                    "blocked_reason": "Exact mounted text absent from RT1 receipt; no parent-text substitution",
                    "source": str(path),
                })

    # Prior art: Scout C5 six adversarial categories (GRM, 2026); contrast
    # sets (Gardner et al., 2020), unverified — lead to check. Ours: literals.
    for family, entity, value in (("S", "atlas tone", "Cobalt-1-India"),
                                  ("N", "Polaris mark", "Marble-4-Juliet")):
        a, digit, c = value.split("-")
        answer = value.replace("-", " ") if family == "S" else value
        source = f"The current {entity.lower()} value is {value}."
        question = f"What is the current {entity} value?"
        controls = [
            ("changed_digit", answer.replace(digit, str(int(digit)+1)), [source]),
            ("omitted_token", f"{a} {c}", [source]),
            ("swapped_relation", f"The {entity} value is {answer}",
             [f"The current {entity} value is Amber-9-Echo. The current {entity} backup value is {value}."]),
            ("negation", f"The {entity} value is not {answer}", [source]),
            ("scattered_word", answer,
             [f"The current {entity} value is Amber-{digit}-Echo. {a} visited yesterday. Travel ends in {c}."]),
            ("alias_collision", f"The {entity} value is {answer}",
             [f"The current {entity.split()[0]} South {entity.split()[-1]} value is {value}."]),
            ("source_negation", answer, [f"The current {entity} value is not {value}."]),
            ("split_across_mounts", answer, [f"The current {entity} value is {a}-{digit}", c+"."]),
            ("extra_token", answer+" Extra", [source]),
            ("source_suffix", answer, [f"The current {entity} value is {value}-Extra."]),
        ]
        for kind, ans, texts in controls:
            fixtures.append({"id": f"control:{family}:{kind}", "class": family,
                             "kind": kind, "question": question, "answer": ans,
                             "texts": texts, "expected": [value], "desired_correct": False,
                             "evidence": "constructed_adversarial_control"})
        fixtures.append({"id": f"diagnostic:{family}:positive", "class": "diagnostic",
                         "question": question, "answer": f"The {entity} value is {answer}",
                         "texts": [source], "expected": [value], "desired_correct": True,
                         "evidence": "constructed_positive_not_a_served_recovery"})

    protected = []
    for folder in ("core", "scripts", "tests", "config", "orders", "docs"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and "grm_c5" not in path.name:
                protected.append(record(path))
    write_new(OUT / "fixtures.json", fixtures)
    write_new(OUT / "protected_files.json", protected)
    cells = [{"id": f"offline:{arm}:{row['id']}", "arm": arm,
              "fixture": row["id"], "device": "CPU", "wall_estimate_s": 0.05}
             for row in fixtures for arm in ("0", "S", "N", "W")]
    registration = {
        "schema": "grm.c5.registration.v1", "order": record(ROOT / "orders/GRM_C5_T30_T33_GROUNDING.md"),
        "fixtures": record(OUT / "fixtures.json"), "protected_files": record(OUT / "protected_files.json"),
        "inputs": list(inputs.values()), "fixture_count": len(fixtures),
        "control_counts": {"S": 10, "N": 10}, "arms": ["0", "S", "N", "W"],
        "default_rule": "0", "normalized": True, "offline_cells": cells,
        "cpu_gate": {"wall_estimate_s": 90, "required_exit_code": 0,
                     "tests": ["tests/test_grm_c5_grounding.py", "tests/test_grm_sc1_1_grounding_glyphs.py",
                               "tests/test_grm_admission.py", "tests/test_grm_eb1_ephemeral_frame.py",
                               "tests/test_grm_rt1_split_child_routing.py"]},
        "mutation_gate": {"wall_estimate_s": 20, "minimum_kill_fraction": 0.8,
                          "mutants": ["default_S", "accept_all", "S_uses_N", "W_uses_baseline", "N_uses_baseline"]},
        "acceptance": {"S_target": "SC1:t30", "N_target": "EB1:t33",
                       "require_baseline_false_target_true_and_expected_value": True,
                       "max_new_false_acceptances_each_arm_all_controls": 0,
                       "max_regressions_on_original_baseline_acceptances": 0,
                       "both_intended_recoveries_required": True,
                       "missing_grounding_inputs_prevent_full_coverage_pass": True},
        "predictions": {"S": "recovers SC1:t30 with zero new control acceptances",
                        "N": "recovers EB1:t33 with zero new control acceptances (order prediction; admission abstention challenges premise)",
                        "W": "at least one NEW scattered_word or alias_collision false acceptance"},
        "rule_spec": {"S": "baseline OR complete ordered three-part value in one affirmative entity-relation source clause; only value separators fold",
                      "N": "baseline OR full proper-name relation and exact hyphenated value in one affirmative source clause; eligible binding helper exposed",
                      "W": "baseline OR global hyphen/whitespace collapse to space followed by unchanged pooled grounding",
                      "0": "imported production grounding with normalized=True; no new rule"},
        "gpu_cells": [], "gpu_budget_hours": 0.4,
        "gpu_condition": "Only after offline verdict change on a probe whose served text is absent; register amendment before any GPU execution",
        "gpu_rails": {"worker_s": 285, "outer_s": 590, "cooldown_s": 30,
                      "lock": "/tmp/forge-gpu.lock", "non_fit": "register non-fit; never extend lease"},
        "claim_scope": "A passing rule is a narrower grounding rule for a named class, not a prose-grounding certificate.",
        "prior_art": "Local SC1.1 / ADM1 / RT1 / Scout (GRM, 2026); Thompson 1968 regex search; Codd 1970 relational keys; Pearl 2000 interventions; Gardner et al. 2020 contrast sets. External references unverified — lead to check. Borrowed mechanisms; ours is restricted grammar and experiment composition.",
        "author_model": "GPT-6 (Codex); exact backend model identifier not exposed to this seat",
        "reasoning_effort": "high",
    }
    write_new(OUT / "registration.json", registration)
    (OUT / "registration.sha256").write_text(digest(OUT / "registration.json") + "  registration.json\n")
    print(json.dumps({"fixture_count": len(fixtures), "offline_cell_count": len(cells),
                      "registration_sha256": digest(OUT / "registration.json")}))


if __name__ == "__main__":
    prepare()
