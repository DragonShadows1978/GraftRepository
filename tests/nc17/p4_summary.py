#!/usr/bin/env python3
"""NC17-P4 summary aggregator (CPU-only). Reads the three gate JSONs and any
bf16 adjudication re-run JSONs, writes logs/nc17/p4_summary.json — the P4
deliverable: gate results, recall + residency tables, INT6-vs-bf16 adjudication
for any INT6 gate that failed."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / "nc17"


def load(name):
    p = LOG / name
    return json.loads(p.read_text()) if p.exists() else None


gates = {
    "A_equivalence": load("p4_gate_a.json"),
    "B_state": load("p4_gate_b.json"),
    "C_e4_arena": load("p4_gate_c.json"),
}
adj = {
    "A_equivalence": load("p4_gate_a_bf16.json"),
    "B_state": load("p4_gate_b_bf16.json"),
    "C_e4_arena": load("p4_gate_c_bf16.json"),
}

summary = {"schema": "nc17_p4_grm_proof_v1",
           "product_engine": "int6-fork",
           "adjudication_engine": "bf16-canon",
           "gates": {}}
for name, g in gates.items():
    if g is None:
        summary["gates"][name] = {"status": "MISSING"}
        continue
    entry = {"int6_pass": bool(g.get("pass")), "int6": g}
    if not g.get("pass"):
        a = adj.get(name)
        if a is not None:
            if a.get("pass"):
                entry["adjudication"] = "QUANT-SENSITIVITY (bf16 PASS, INT6 FAIL)"
            else:
                entry["adjudication"] = "GRM-MACHINERY (bf16 also FAIL)"
            entry["bf16"] = a
        else:
            entry["adjudication"] = "PENDING (bf16 re-run not present)"
    summary["gates"][name] = entry

if gates["C_e4_arena"]:
    c = gates["C_e4_arena"]
    summary["recall_table"] = c["recall"]
    summary["residency_table"] = c["residency"]
if gates["A_equivalence"]:
    a = gates["A_equivalence"]
    summary["equivalence"] = {"max_dlogit": a["a1_equivalence"]["max_dlogit"],
                              "hard_flips": a["a1_equivalence"]["hard_flips"]}
if gates["B_state"]:
    b = gates["B_state"]
    summary["state"] = {"max_dlogit": b["max_dlogit"],
                        "bit_identical": b["bit_identical"]}

summary["all_gates_pass_int6"] = all(
    g is not None and g.get("pass") for g in gates.values())
(LOG / "p4_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"\n[summary] wrote {LOG / 'p4_summary.json'}")
