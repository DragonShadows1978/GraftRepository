#!/usr/bin/env python3
"""Author the GRM-C3 immutable registration, before any gates or model work.

Prior art: local GRM-DET1/SC2 (GraftRepository contributors, 2026) supplies
planted misses and carried strict thresholds. This campaign adds new text,
balanced controlled mounts and a frozen 1e-8 structural threshold; no new
detector algorithm is claimed. External priority: unverified — lead to check
"attention mass missing knowledge detection" and "selective prediction".
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/grm_c3"


def digest(value):
    # Prior art: NIST SHA-256 (FIPS 180-4, 2015); canonical JSON is local
    # sorted-key UTF-8, not a claim to implement RFC 8785.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    config = json.loads((ROOT / "config/grm_demand_registered.json").read_text())
    cases = []
    # Prior art: balanced blocked experiments (Fisher, 1935), unverified —
    # lead to check "Fisher Design of Experiments randomized blocks".
    # Here cell order rotates deterministically; it is NOT randomized.
    names = ["Averin", "Belora", "Cadrin", "Dovela", "Eshrin", "Falyra",
             "Gavren", "Hesora", "Ivrali", "Jaspen", "Kovira", "Lurien"]
    properties = ["beacon color", "cabinet material", "signal shape", "banner color",
                  "cradle material", "marker shape", "panel color", "case material",
                  "badge shape", "lamp color", "tray material", "seal shape"]
    answers = ["vermilion", "titanium", "hexagon", "indigo", "ceramic", "crescent",
               "ochre", "nickel", "triangle", "magenta", "copper", "diamond"]
    wrong = ["turquoise", "aluminum", "pentagon", "scarlet", "rubber", "square",
             "violet", "leather", "circle", "amber", "glass", "oval"]
    weak = ["482-17-936", "Neravel", "731-06-284", "Ostelin", "926-43-815", "Pheradis",
            "365-82-147", "Quenivar", "814-59-263", "Rovethan", "597-24-681", "Seluvin"]
    classes = ["correct", "decoy", "unavailable", "weak"]
    for i, name in enumerate(names):
        for cls in classes:
            entity = f"{name} {dict(correct='observatory', decoy='relay', unavailable='archive', weak='annex')[cls]}"
            prop = properties[i] if cls != "weak" else ("access number" if i % 2 == 0 else "custodian name")
            value = weak[i] if cls == "weak" else answers[i]
            bad = ("999-88-777" if i % 2 == 0 else "Talomer") if cls == "weak" else wrong[i]
            fact = f"The {entity} {prop} is {value}."
            # Decoy is a topical sibling relation, not a contradictory true
            # answer to the same question. Oracle target never enters its repo.
            decoy = f"The {entity} training {prop} is {bad}."
            deposited = decoy if cls == "decoy" else fact
            node_id = f"c3_{i+1:02d}_{cls}_node"
            nodes = [] if cls == "unavailable" else [{
                "node_id": node_id, "text": f"User: Remember this record.\nAssistant: {deposited}",
                "supersedes": [], "role": "decoy" if cls == "decoy" else "target"}]
            case = {"id": f"c3_{i+1:02d}_{cls}", "class": cls, "entity": entity,
                    "question": f"What is the {entity} {prop}? Reply with only the value.",
                    "oracle_fact": fact, "expected_values": [value],
                    "wrong_fact_values": [bad], "stale_values": [], "nodes": nodes,
                    "mount_node_ids": [] if not nodes else [node_id],
                    "weakness_target": ("separator_number" if i % 2 == 0 else "proper_name") if cls == "weak" else None}
            case["sha256"] = digest(case)
            cases.append(case)
    cells = []
    for i in range(12):
        group = cases[i*4:i*4+4]
        group = group[i % 4:] + group[:i % 4]
        cells.append({"id": f"trace_{i+1:02d}", "case_ids": [c["id"] for c in group],
                      "estimate_seconds": 200, "worker_cap_seconds": 225,
                      "outer_cap_seconds": 590, "cooldown_seconds": 30,
                      "estimate_basis": "reasoning only: four fresh processes-in-sequence model loads at 25s plus 25s planting/decode/cleanup each; no live C3 timing"})
    registration = {
        "schema": "grm.c3.registration.v1", "order": "orders/GRM_C3_DNGH_DECOY_CALIBRATION.md",
        "order_sha256": hashlib.sha256((ROOT / "orders/GRM_C3_DNGH_DECOY_CALIBRATION.md").read_bytes()).hexdigest(),
        "cases": cases, "class_counts": dict.fromkeys(classes, 12),
        "case_set_sha256": digest(cases),
        "rules": {"carried": {"threshold": config["threshold"], "comparison": "lt"},
                  "candidate": {"threshold": config["candidate_threshold"], "comparison": "lt"},
                  "near_zero": {"threshold": 1e-8, "comparison": "lt"},
                  "empty_band": {"comparison": "mounted_token_count_eq_zero"}},
        "threshold_caveat": config["caveat"],
        "near_zero_rationale": "fixed numerical near-zero tolerance 1e-8, chosen before traces, never refit; exact 0 fires and equality 1e-8 does not",
        "acceptance": {"miss_detection_min": 0.9, "false_fire_max": 0.05,
                       "tp_min_of_24": 22, "fp_max_of_24": 1, "lost_correct_answers_max": 0},
        "prediction": ["near_zero fires on empty bands but misses at least one nonempty decoy",
                       "carried causes avoidable false fires on weak correct reads"],
        "qualification": "all 48 retained; target/decoy mount and oracle absence checked; correct and weak must answer correctly for achieved class qualification. Weakness is targeted, not guaranteed by fixture text. No replacement or filtering.",
        "preservation": "count rule fires on every actually correct OFF answer as correct_answers_at_risk; require zero for conservative offline eligibility. Actual demand-ON lost answers are UNMEASURED; no production flip certification from OFF traces.",
        "design": "controlled mount intervention with production _attempt and EB1 clean live state, not natural routing or a chronological-session replication; empty-band means actual seated token count zero",
        "trace": {"ngen": 32, "detector_on": False, "observer": "D-NGH only, full-layer mean mounted mass, every generated prediction including stop tokens, unused final flush dropped"},
        "model": {"id": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee", "generation": "greedy, no explicit reasoning-effort override; inherited Harmony prompt"},
        "agent": {"model": "GPT-6 (exact backend variant not exposed)", "requested_effort": "high"},
        "cells": cells, "gpu_budget_seconds": 2700,
        "stop": "225s child timeout => terminal NON_FIT, stop campaign; never retry, extend, replace or reuse partial results for acceptance; any other child failure stops campaign RED",
        "cpu_gates": ["fixture validity and novelty", "hand-computed trace table and strict boundaries", "24/24 acceptance integer boundaries", "malformed/incomplete receipts rejected", "observer adapter fake-arena one-answer call", "dry-run all cells"],
        "prior_art": "GRM-DET1/SC2/SC1.2, GraftRepository contributors 2026: import planting, observer, scoring and strict crossing/minimum helpers. Fisher 1935 blocked design and NIST 2015 SHA-256 are leads; external references unverified — lead to check. New: fixture text, balanced controlled-mount protocol, offline comparison and safety wrapper; no detector novelty claimed."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "registration.json"
    with path.open("x") as stream:
        json.dump(registration, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with (OUT / "registration.sha256").open("x") as stream:
        stream.write(f"{sha}  artifacts/grm_c3/registration.json\n")
    print(sha)


if __name__ == "__main__":
    main()
