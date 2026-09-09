#!/usr/bin/env python3
"""Freeze X1 BEFORE gates. Prior art: house SUP/census fixtures (2026),
Fisher, The Design of Experiments (1935; unverified — lead to check): paired
blocked comparisons. New: this fixed GRM multiplicity/absence comparison.
No model imports; selected census constructors are read by AST.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/grm_x1"


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def census():
    path = ROOT / "scripts/grm_e2e_session.py"
    tree = ast.parse(path.read_text())
    names = {"plant_acceptance", "fact_turn", "supersede_turn", "filler_turn",
             "probe_turn", "build_full_script"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    # Existing literal fixture constructors only; no module imports or CUDA.
    module = ast.Module(body=selected, type_ignores=[])
    scope = {"Any": object}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["build_full_script"]()


def freeze():
    if (OUT / "registration.json").exists():
        raise SystemExit("registration exists: immutable; amendments must be separate")
    fixtures = OUT / "fixtures"
    families = {}
    sources = ["orders/GRM_X1_VIRTUAL_ADDRESSES.md", "docs/GRM_SCOUT_2026-09-08.md",
               "config/grm_live_registered_baselines.json", "config/grm_demand_registered.json",
               "scripts/grm_e2e_session.py", "scripts/grm_x1_register.py"]
    sup_entities = {"lumen": "seal", "harbor": "token", "orion": "pin",
                    "praxis": "dock", "solace": "key"}
    for path in sorted((ROOT / "tests/fixtures/supersession_battery").glob("*.json")):
        sources.append(str(path.relative_to(ROOT)))
        original = json.loads(path.read_text())
        create(fixtures / "sources" / path.name, path.read_bytes())
        for probe in original["probes"]:
            entity = probe["target_node"].split("_")[0]
            relation = sup_entities[entity]
            family_id = "sup_" + entity
            versions = list(probe["stale_nodes"]) + [probe["target_node"]]
            nodes = []
            for node in original["nodes"]:
                row = dict(node)
                if row["node_id"] in versions:
                    row["address"] = [entity, relation]
                    row["version"] = versions.index(row["node_id"]) + 1
                else:
                    # Oracle metadata is fixture-owned, never inferred from answers.
                    row["address"] = [row["node_id"], "independent_fact"]
                    row["version"] = 1
                nodes.append(row)
            families[family_id] = {"family_id": family_id, "source": str(path.relative_to(ROOT)),
                "nodes": nodes, "target_node": probe["target_node"],
                "decoy_node": probe["competitor_node"], "address": [entity, relation],
                "expected": probe["expected_values"][0],
                "rejected": probe["stale_values"] + probe["wrong_fact_values"]}
    events = census()
    create(fixtures / "sources/census_events.json", raw_json(events))
    for entity, relation in [("orion", "pin"), ("lyra", "dock"), ("cypher", "bridge"),
                             ("nova", "key"), ("mira", "seal"), ("terra", "port"),
                             ("ember", "code"), ("atlas", "tone")]:
        selected = [(i, e) for i, e in enumerate(events)
                    if e.get("fact_id") == f"{entity} {relation}"
                    and e["kind"] in ("fact", "supersede")]
        nodes = []
        for version, (i, event) in enumerate(selected, 1):
            nodes.append({"node_id": f"census_t{i:02d}", "role": event["kind"],
                "value": event["value"], "text": f"User: {event['user']}\nAssistant: {event['assistant']}",
                "supersedes": [nodes[-1]["node_id"]] if nodes else [],
                "address": [entity, relation], "version": version})
        # Existing SUP competitor text, unchanged: no synthetic answer-bearing decoy.
        decoy = dict(families["sup_orion"]["nodes"][-1])
        nodes.append(decoy)
        family_id = "census_" + entity
        families[family_id] = {"family_id": family_id, "source": "scripts/grm_e2e_session.py::build_full_script",
            "nodes": nodes, "target_node": nodes[-2]["node_id"], "decoy_node": decoy["node_id"],
            "address": [entity, relation], "expected": nodes[-2]["value"].casefold(),
            "rejected": [n["value"].casefold() for n in nodes[:-2]] + [decoy["value"].casefold()]}
    alias_families = ["sup_lumen", "sup_harbor", "sup_orion", "sup_praxis", "sup_solace",
                      "census_lyra", "census_cypher", "census_nova", "census_mira",
                      "census_terra", "census_ember", "census_atlas"]
    relation_alias = {"seal": "sealing identifier", "token": "access token", "pin": "PIN identifier",
                      "dock": "docking identifier", "key": "access key", "bridge": "bridge identifier",
                      "port": "port identifier", "code": "access code", "tone": "tone identifier"}
    queries = []
    for family_id in alias_families:
        family = families[family_id]
        entity, relation = family["address"]
        queries.append({"query_id": f"alias_{len(queries):02d}", "kind": "alias", "family_id": family_id,
            "question": f"For {entity.upper()}, give the {relation_alias[relation]} currently on record.",
            "oracle_address": family["address"], "expected": family["expected"], "rejected": family["rejected"]})
    for family_id in ["sup_lumen", "sup_harbor", "sup_orion", "census_orion", "census_lyra", "census_mira"]:
        family = families[family_id]
        entity, relation = family["address"]
        for question in [f"After all recorded revisions, what {relation} should I use for {entity.title()}?",
                         f"Give the latest authoritative {entity.title()} {relation}; disregard its earlier versions."]:
            queries.append({"query_id": f"correction_{len(queries)-12:02d}", "kind": "correction",
                "family_id": family_id, "question": question, "oracle_address": family["address"],
                "expected": family["expected"], "rejected": family["rejected"]})
    aliases = {"entities": {f["address"][0]: [f["address"][0]] for f in families.values()},
               "relations": {r: [r, a] for r, a in relation_alias.items()}}
    create(fixtures / "families.json", raw_json(families))
    create(fixtures / "queries.json", raw_json(queries))
    create(fixtures / "aliases.json", raw_json(aliases))
    cells = [{"cell": f"oracle_m{m}_s{s}", "phase": "oracle", "multiplicity": m,
              "query_ids": [q["query_id"] for i, q in enumerate(queries) if i % 2 == s],
              "arms": ["A", "B", "C"], "conditions": ["present", "absent"], "worker_cap_s": 285}
             for m in (1, 10, 100) for s in (0, 1)]
    cells += [{"cell": f"natural_m{m}", "phase": "natural", "multiplicity": m,
               "query_ids": [q["query_id"] for q in queries], "arms": ["N"],
               "conditions": ["present", "absent"], "worker_cap_s": 285} for m in (1, 10, 100)]
    create(fixtures / "cells.json", raw_json(cells))
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sorted(fixtures.rglob("*.json"))}
    create(fixtures / "SHA256SUMS", "".join(f"{v}  {k}\n" for k, v in manifest.items()).encode())
    registration = {
        "schema": "grm.x1.registration.v1", "evidence_class": "reasoning: pre-gate registration",
        "order": sources[0], "created_utc": "2026-09-08", "source_shas": {p: sha(ROOT / p) for p in sources},
        "fixture_shas": manifest, "fixture_manifest_sha": sha(fixtures / "SHA256SUMS"),
        "author": {"model": "GPT-6 (system identity; exact deployment ID unavailable)",
                   "reasoning_effort": "not exposed; not inferred"},
        "lead_predictions": {"P1": "A present exact(1)-exact(100) >= 0.20",
            "P2": "B present max exact-min exact across 1,10,100 <= 0.05",
            "P3": "C all-condition false(100) <= 0.5*A all-condition false(100); requires A false > 0",
            "P4": "B/A and C/A mean wall-per-turn ratios at each multiplicity in [0.90,1.10]"},
        "seat_predictions": {"S1": "A present decline < 0.20: existing RT1 may already resist duplicate children",
            "S2": "B present exact spread <= 0.05, but correctly mounted read errors may persist",
            "S3": "C absent false rate = 0 and structural abstention correctness = 1; no generation on fault",
            "S4": "B/C mean wall no more than 1.10*A; C may be faster than the P4 lower bound",
            "S5": "N present exact >= C present exact-0.05, limited to the frozen dictionary aliases"},
        "frame": {"model_id": "openai/gpt-oss-20b", "reasoning_effort": "low (existing Harmony sink)",
            "model_dir": "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
            "native_lib": "/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so",
            "ephemeral": True, "arena_width": 96, "topk": 3, "max_trips": 2, "ngen": 24,
            "storage_bits": 8, "capture_pin": "live", "seat_near_live": True,
            "sup_resolve": True, "adm_decisive": True, "lsr_fixes": True, "demand_ngh": False,
            "demand_note": "Production D-NGH default OFF, threshold config frozen unchanged. Count structural X1 demands separately; no attention telemetry."},
        "fixture_contract": {"queries": 24, "alias": 12, "correction": 12, "families": len(families),
            "held_out": "New query wording, frozen before tests; existing values/documents are NOT held-out entities or unseen facts. No fitting on these queries.",
            "multiplicity": [1, 10, 100],
            "decoys": "Exactly m additional children per target's existing competitor family; cycle native width-split children in source order. Original children remain. Shared immutable K/V backing, distinct IDs and metadata; no new harvest for duplicates.",
            "absence": "Same query, remove ALL target-current-version pages from eligible mounted set. Keep oracle version pointer and stale/decoy pages. Also CPU-test unknown address and partial page sets.",
            "pairing": "seed 20260908; balanced rotation A/B/C by query index and condition; reset repository/cache and RNG before every arm from shared native payload snapshot; no query answer deposits",
            "census_decoy_limit": "Census controls reuse existing Falcon competitor verbatim; it does not name every census target. Report source-stratified results."},
        "arms": {"A": "Unmodified EB1 arena.step, addresses OFF",
            "B": "Exact fixture address -> current pages -> native _attempt; absent/invalid mount falls back to A",
            "C": "B with structural PAGE_FAULT abstention before generation on absent/invalid mount",
            "N": "Conditional C with frozen unique token-span dictionary resolver; no per-query oracle lookup"},
        "scoring": {"exact": "Full answer strip, casefold, normalized Unicode dash/whitespace, remove outer quotes and terminal . only; equality to expected value. Present denominator only.",
            "false_answer": "Any non-abstaining non-exact present answer, or any non-abstaining absent answer. Conservative: verbose correct answers still fail strict exact; record value-match diagnostic separately.",
            "abstention": "Structural X1 fault, production abstain info, or registered full-string refusal vocabulary; no substring refusal escape",
            "abstention_correctness": "correct abstentions/all abstentions; null when none. Also report absent recall and present false abstention rate",
            "wall": "GPU synchronize outside/inside timed turn boundaries; exclude reset, install, load, scoring. Raw per-turn rows plus per-condition and all-condition means",
            "demand": "structural address misses counted separately from D-NGH demand_fired (OFF); no fabricated telemetry"},
        "rails": {"oracle_positive": "All 6 cells complete, B present min exact >=0.80, spread<=0.05, B100-A100>=0.20, C absent false=0, C present>=B present-0.05 at every m, B/C present mount address coverage=1",
            "kill": "With exact addresses: selection depends on decoy ranking (coverage changes/missing under present) OR B present spread>0.05. Natural: N loses >0.05 exact or advantage over A100 <0.20. Strict empirical rail; thin paired sample is not a universal refutation.",
            "natural_unlock": "Only oracle_positive; no override. If oracle null/negative, natural is NOT RUN.",
            "failures": "Any worker failure/incomplete receipt stops resume; no automatic retry or new fingerprint campaign. Amendment/lead decision required.",
            "cpu": ["X1 invariant tests", "existing RT1 and RS3 frame tests", "fixture hash/content checks", "dry-run every cell", "mutants killed >=0.80 after passing baseline"],
            "blind_verification": "Lead dispatch required; author does not select or launch verifier"},
        "budget": {"primary_cells": 6, "conditional_cells": 3, "worker_cap_s": 285,
            "primary_gpu_s_max": 1710, "all_gpu_s_max": 2565, "order_gpu_s_max": 2700,
            "outer_cap_s": 590, "lock_wait_s": 250, "cooldown_s": 30,
            "resume": "one next eligible cell per foreground invocation; never an unbounded loop",
            "timeout": "Worker self SIGALRM raises TimeoutError; no external signals or kill. Non-returning native calls cannot be hard-preempted without violating never-kill; residual explicitly RED."},
        "prior_art": {"Codd 1970": "logical relational identity; https://doi.org/10.1145/362384.362685; bibliography checked via ACM SIGMOD",
            "Denning 1968": "bounded working sets/page absence; https://denninginstitute.com/pjd/PUBS/WSModel_1968.pdf (primary source checked)",
            "Reed 1978": "multiversion indirection; unverified — lead to check: Naming and Synchronization in a Decentralized Computer System",
            "Fisher 1935": "blocked paired experiment; unverified — lead to check: The Design of Experiments",
            "Unicode UAX15": "NFKC normalization; unverified — lead to check: Unicode normalization annex 15",
            "house 2026": "reuse EB1/SUP/RT1/RS3/RS4 and census; new work is additive address contract and this registered comparison, no novelty claim"},
    }
    create(OUT / "registration.json", raw_json(registration))
    create(OUT / "registration.sha256", (sha(OUT / "registration.json") + "  artifacts/grm_x1/registration.json\n").encode())
    print(json.dumps({"registration_sha256": sha(OUT / "registration.json"), "fixtures": manifest}, indent=2))


if __name__ == "__main__":
    freeze()
