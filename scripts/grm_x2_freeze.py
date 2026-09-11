#!/usr/bin/env python3
"""Create-only GRM-X2 preregistration and fixtures; never imports the checker.

Prior art: house DET1/EB1 (2026) frozen fixtures and paired gates; Fisher
(1935), controlled comparisons, unverified — lead to check: Fisher Design
of Experiments 1935. Ours: this deterministic adversarial construction and
the six-cell schedule, not a new experimental-design method.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/grm_x2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as f:
        f.write(raw)
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(raw), "bytes": len(raw)}


def source(sid, text):
    return {"source_id": sid, "source_version": digest(text.encode()), "text": text}


def main():
    if (OUT / "registration.json").exists():
        raise SystemExit("Registration exists; immutable, use a separate amendment.")
    facts = [
        ("Orion", "owner", "Mira", "Neri", "Mira owns Orion."),
        ("Lyra", "location", "Delft", "Leiden", "Lyra is based in Delft."),
        ("Cypher", "color", "amber", "violet", "Cypher is colored amber."),
        ("Nova", "manager", "Mira", "Neri", "Nova is managed by Mira."),
        ("Terra", "supplier", "Acme", "Rivet", "Acme supplies Terra."),
        ("Solace", "capital", "Lima", "Quito", "Lima is the capital of Solace."),
        ("Harbor", "token", "Nacre-6-Blue", "Opal-7-Green", "For Harbor, the token is Nacre-6-Blue."),
        ("Praxis", "dock", "Quartz-8-Jade", "Raven-9-Ivory", "Praxis uses Quartz-8-Jade as its dock."),
        ("Ember", "key", "Birch-2-Beacon", "Auric-4-Alpha", "Ember's key reads Birch-2-Beacon."),
        ("Atlas", "code", "Cobalt-1-India", "Silver-6-Charlie", "The code assigned to Atlas is **Cobalt‑1‑India**."),
    ]
    cpu = []
    for n, (entity, relation, value, other, paraphrase) in enumerate(facts):
        sibling = "Tundra"
        question = f"What is the {relation} of {entity}?"
        ordinary = f"The {relation} of {entity} is {value}. The {relation} of {sibling} is {other}."
        distract_relation = "location" if relation != "location" else "owner"
        scattered = f"The {distract_relation} of {entity} is {value}. The {relation} of {sibling} is {other}."
        for kind, answer, text, correct in [
            ("role_swapped", f"The {relation} of {entity} is {other}.", ordinary, False),
            ("negated", f"The {relation} of {entity} is not {value}.", ordinary, False),
            ("scattered_token", f"The {relation} of {entity} is {other}.", scattered, False),
            ("correct_paraphrase", paraphrase, ordinary, True),
        ]:
            cpu.append({"id": f"{kind}_{n:02}", "kind": kind, "question": question,
                        "answer": answer, "correct": correct,
                        "sources": [source(f"cpu_{kind}_{n:02}", text)]})
    # Reuse the actual pure fixture builders, AST-selected without GPU imports.
    fixture_script = ROOT / "scripts/grm_e2e_session.py"
    tree = ast.parse(fixture_script.read_text())
    names = {"plant_acceptance", "fact_turn", "supersede_turn", "filler_turn", "probe_turn", "build_full_script"}
    env = {}
    pure = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] +
                      [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names], type_ignores=[])
    exec(compile(ast.fix_missing_locations(pure), str(fixture_script), "exec"), env)
    events = env["build_full_script"]()
    selected = [x for x in events if x["kind"] == "fact"][:8]
    sup_path = ROOT / "tests/fixtures/supersession_battery/fresh_fact_controls.json"
    sup = json.loads(sup_path.read_text())
    single = []
    for event in selected:
        entity, rel = event["fact_id"].rsplit(" ", 1)
        single.append({"entity": entity, "relation": rel, "value": event["value"],
                       "question": f"What is the current {event['fact_id']} value?",
                       "source": source(f"eb1_{entity}_{rel}", event["user"]),
                       "origin": "scripts/grm_e2e_session.py::build_full_script initial fact"})
    for node, probe in zip(sup["nodes"][:2], sup["probes"]):
        entity, rel = ("Praxis", "dock") if node["node_id"] == "praxis_fact" else ("Solace", "key")
        single.append({"entity": entity, "relation": rel, "value": node["value"],
                       "question": probe["question"], "source": source(node["node_id"], node["text"]),
                       "origin": str(sup_path.relative_to(ROOT))})
    live = []
    for i, row in enumerate(single):
        live.append({"id": f"correct_{i:02}", "mount_class": "correct", "question": row["question"],
                     "expected": [row["value"]], "expected_bindings": [{k: row[k] for k in ("entity", "relation", "value")}],
                     "sources": [row["source"]], "origin": row["origin"]})
    for n, (a, b) in enumerate(((single[8], single[9]), (single[0], single[3]))):
        live.append({"id": f"correct_join_{n}", "mount_class": "correct",
                     "question": f"What are the current {a['entity']} {a['relation']} and {b['entity']} {b['relation']} values?",
                     "expected": [a["value"], b["value"]],
                     "expected_bindings": [{k: row[k] for k in ("entity", "relation", "value")} for row in (a, b)],
                     "sources": [a["source"], b["source"]], "origin": "two-source conjunction of frozen EB1/sup facts"})
    for i, row in enumerate(single[:6]):
        text = (f"Index context mentions {row['entity']} {row['relation']}. "
                f"The current Tundra {row['relation']} value is Sable-0-Copper. "
                f"The current {row['entity']} color value is amber.")
        live.append({"id": f"decoy_{i:02}", "mount_class": "decoy", "question": row["question"],
                     "expected": [], "expected_bindings": [], "withheld_truth": row["value"],
                     "sources": [source(f"decoy_{i}", text)], "origin": "supersession competitor pattern: mentioned entity, wrong relation"})
    for i, row in enumerate(single[:6]):
        live.append({"id": f"unavailable_{i:02}", "mount_class": "unavailable", "question": row["question"],
                     "expected": [], "expected_bindings": [], "withheld_truth": row["value"], "sources": [],
                     "origin": "EB1 fact withheld, empty repository"})
    inputs = ["orders/GRM_X2_RELATIONAL_WITNESSES.md", "core/graft_arena.py", "core/grm_text_norm.py", "core/grm_frame.py",
              "scripts/grm_e2e_session.py", "config/grm_live_registered_baselines.json", "config/grm_demand_registered.json",
              "tests/fixtures/supersession_battery/fresh_fact_controls.json"]
    input_records = [{"path": p, "sha256": digest((ROOT / p).read_bytes())} for p in inputs]
    frozen = [put(OUT / "fixtures/cpu.json", cpu), put(OUT / "fixtures/live.json", live)]
    reg = {
        "schema": "grm.x2.registration.v1", "date": "2026-09-08", "order": inputs[0], "immutable": True,
        "evidence_class": "preregistered design; no gate run", "flag": {"name": "GRM_X2_WITNESSES", "default": "OFF", "unknown": "OFF"},
        "lead_predictions": {"P1": "A accepts >=60% of 20 role-swapped+negated answers", "P2": "B rejects >=90% of those 20 and accepts >=90% of 10 correct paraphrases", "P3": "B decoy false-answer rate <=0.5*A; B rejects <=1 of 12 correct-mount answers"},
        "seat_predictions": {"P1": {"expected_accept_fraction": 1.0, "confidence": 0.95},
                             "P2": {"expected_bad_reject_fraction": 1.0, "expected_paraphrase_accept_fraction": 0.9, "confidence": 0.60},
                             "P3": {"expected_decoy_false_answer_rate_B": 0.0, "expected_correct_answers_rejected": 2,
                                    "confidence": 0.55, "prediction": "False answers fall, but conservative parsing may FAIL the <=1 rejection rail"}},
        "cpu": {"n": 40, "per_class": 10, "positive": "correct supported answer", "strict_full_answer_coverage": True,
                "kill_if": "bad rejection <0.9 OR correct paraphrase acceptance <0.9; do not tune after gate; GPU blocked on kill"},
        "implementation": "new opt-in per-arena deposit adapter + explicit answer gate; no edits to existing core; metadata only; unsupported assertion rejects; no learned extractor",
        "gpu": {"model_id": "openai/gpt-oss-20b", "width": 96, "ephemeral": True, "capture_pin": "live", "seat_near_live": True,
                "rt1": True, "demand_ngh": False, "ngen": 64, "worker_seconds": 285, "outer_seconds": 590,
                "lock_wait_seconds": 240, "cooldown_seconds": 30, "cells": [f"cell_{i:02}" for i in range(6)],
                "questions_per_cell": 4, "total_questions": 24, "arms": ["A", "B"],
                "comparison": "same generated bytes/mounts, A existing grounding; B A AND witness; rejection structural abstention",
                "mount_intervention": "fresh EB1 repository for each question, explicit correct/decoy/empty mounts, production fit and _attempt; router diagnostic with RT1 on; not natural-routing efficacy",
                "budget_seconds": 1800, "reserved_worker_seconds": 1710, "common_order_cap_seconds": 2700,
                "retry": "no automatic retry; failed/started cells consume full reserved budget; amendment needed",
                "score": "exact semantic value(s) with complete binding coverage; false non-abstention; abstention correctness; A-correct answers B rejects; per class denominators",
                "unavailable_and_decoy": "abstention required because requested binding has no mounted witness; lucky unsupported truth is reported separately"},
        "mutations": ["ignore_entity", "ignore_relation", "ignore_polarity", "skip_unknown_clause", "ignore_source_version"],
        "mutation_pass_fraction": 0.8, "blind_red_team": "lead dispatch required; author unit tests are baseline only",
        "fixtures": frozen, "input_records": input_records,
        "prior_art": ["Fader, Soderland, Etzioni 2011 ReVerb: relation triples and extraction constraints; ours is smaller hand grammar",
                      "Buneman, Khanna, Tan 2001 Why and Where: source-location provenance; ours source version and exact offsets beside native KV",
                      "Green, Karvounarakis, Tannen 2007 Provenance Semirings: explicit conjunction of evidence; ours simple join receipts, no semiring implementation",
                      "House SC1.1/EB1/RS3/RS4/RT1 (2026): glyph normalization, frame, capture/seating and routing reused"],
        "model_seat": {"id": "GPT-6 (system identity; precise backend variant unavailable)", "reasoning_effort": "not exposed by runtime"},
    }
    record = put(OUT / "registration.json", reg)
    put(OUT / "fixture_manifest.json", {"registration": record, "fixtures": frozen})
    print(json.dumps({"registration": record, "fixtures": frozen}, indent=2))


if __name__ == "__main__":
    main()
