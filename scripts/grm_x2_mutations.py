#!/usr/bin/env python3
"""Five preregistered mutants in copies; production files never overwritten.

Prior art: DeMillo, Lipton & Sayward (1978), mutation testing, unverified —
lead to check: Hints on Test Data Selection 1978. House §8 (2026) requires
>=0.80 non-error kills after a passing baseline. Ours: concrete X2 operators.
"""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_x2_common import OUT, contract, create_json, sha_file, read


def run():
    contract()
    if read(OUT / "cpu_gate.json")["status"] != "PASS":
        raise RuntimeError("mutation testing requires passing frozen baseline")
    path = ROOT / "core/grm_x2_witnesses.py"
    original_sha = sha_file(path)
    original = path.read_text()
    mutations = {
        "ignore_entity": ("return Binding(entity, relation, value, bool(d.get(\"neg\"))), spans", "return Binding('orion', relation, value, bool(d.get(\"neg\"))), spans"),
        "ignore_relation": ("return Binding(entity, relation, value, bool(d.get(\"neg\"))), spans", "return Binding(entity, 'owner', value, bool(d.get(\"neg\"))), spans"),
        "ignore_polarity": ("return Binding(entity, relation, value, bool(d.get(\"neg\"))), spans", "return Binding(entity, relation, value, False), spans"),
        "skip_unknown_clause": ("if unknown or not bindings:", "if not bindings:"),
        "ignore_source_version": ("if (version != sha(source[\"text\"])", "if False and (version != sha(source[\"text\"])")}
    results = []
    directory = OUT / "mutants"
    directory.mkdir(exist_ok=False)
    for name, (old, new) in mutations.items():
        if original.count(old) != 1:
            raise RuntimeError(f"mutation site moved: {name}")
        target = directory / f"{name}.py"
        with target.open("x") as f:
            f.write(original.replace(old, new))
        spec = importlib.util.spec_from_file_location(f"grm_x2_mutant_{name}", target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            from core.grm_x2_witnesses import META
            # The tests' independent attacks are applied to the mutated
            # deposit AND verification code, as they would be in production.
            src = "The owner of Orion is Mira."
            answer = "The owner of Orion is Mira."
            if name == "ignore_entity":
                src, answer = "The owner of Lyra is Mira.", "Mira owns Orion."
            elif name == "ignore_relation":
                src = "The color of Orion is Mira."
            elif name == "ignore_polarity":
                answer = "The owner of Orion is not Mira."
            elif name == "skip_unknown_clause":
                answer = "The owner of Orion is Mira. Penguins can fly."
            meta = module.extract(src, "a")
            if name == "ignore_source_version":
                meta["source_version"] = "0" * 64
            observed = module.verify(answer, "What is the owner of Orion?", [{"text": src, "metadata": {META: meta}}])
            # A mutant is killed when this rejection invariant fails. It is
            # NOT killed merely because compilation/runtime raised an error.
            results.append({"name": name, "killed": bool(observed["accepted"]), "error": None,
                            "mutation_sha256": sha_file(target), "observed": observed})
        except Exception as exc:
            results.append({"name": name, "killed": False, "error": f"{type(exc).__name__}: {exc}", "mutation_sha256": sha_file(target)})
    assert sha_file(path) == original_sha
    n = sum(r["error"] is None for r in results)
    kills = sum(r["killed"] for r in results)
    payload = {"evidence_class": "unit test / author mutation baseline, not blind verification", "results": results,
               "non_error_mutants": n, "killed": kills, "kill_fraction": kills / n if n else 0,
               "status": "PASS" if n == 5 and kills / n >= .8 else "RED", "production_sha256_unchanged": original_sha}
    create_json(OUT / "mutation_gate.json", payload)
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
