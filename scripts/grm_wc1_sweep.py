"""GRM-WC1 registration and pure sweep accounting; no model imports."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/grm_wc1"
MAIN = Path("/mnt/ForgeRealm/GraftRepository")
WIDTHS = (64, 96, 128, 192, 256)
BASELINE = 96
BATTERIES = ("sup", "census", "longhorizon")
FRAME = MAIN / "artifacts/grm_det1/run_20260831T160525Z_2/runtime_frame_28b3196f8fb04a41.json"
CENSUS = FRAME.parent / "det1_11/census/lived_serving_census_98ef71e88dec17a1.json"
FLAGS = {"GRM_PERSISTENT_BOAT": "0", "GRM_LSR_FIXES": "1",
         "GRM_SEAT_NEAR_LIVE": "1", "GRM_CAPTURE_PIN": "live",
         "GRM_DEMAND_NGH": "0", "GRM_RT1_RULE": "1"}


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def write(path, payload, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return record(path)


def literal_constants(path, names):
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError(f"missing source constants: {set(names) - set(result)}")
    return result


def plan(widths=WIDTHS):
    if len(set(widths)) != len(widths) or set(widths) != set(WIDTHS):
        raise ValueError("the registered width grid must occur exactly once")
    return [{"width": w, "battery": b, "gate": "G2" if w == BASELINE else "G3"}
            for w in (BASELINE, *(w for w in widths if w != BASELINE))
            for b in BATTERIES]


def regressions(rows, baseline):
    def index(items):
        result = {r["probe_id"]: r for r in items}
        if len(result) != len(items):
            raise ValueError("duplicate probe id")
        return result
    current, base = index(rows), index(baseline)
    if current.keys() != base.keys():
        raise ValueError("incomplete or mismatched probe set")
    return sorted(k for k in base if base[k]["correct"] and not current[k]["correct"])


def assemble(results):
    indexed = {(r["width"], r["battery"]): r for r in results}
    if len(indexed) != len(results):
        raise ValueError("duplicate battery result")
    if set(indexed) - {(w, b) for w in WIDTHS for b in BATTERIES}:
        raise ValueError("unregistered battery or width")
    table = []
    for cell in plan():
        w, b = cell["width"], cell["battery"]
        result = indexed.get((w, b))
        if result is None:
            table.append({**cell, "status": "NOT_RUN", "correct": None,
                          "total": None, "regressions_vs_96": None,
                          "split_parents": None, "split_children": None,
                          "mean_resident_seats_per_turn": None,
                          "mean_wall_ms_per_turn": None,
                          "mean_mounted_mass_at_readout": None, "turn_count": 0})
            continue
        rows = result["probes"]
        turns = result.get("turns", [])
        base = indexed.get((BASELINE, b))
        def avg(key):
            values = [t[key] for t in turns if t.get(key) is not None]
            return mean(values) if values else None
        table.append({**cell, "status": "MEASURED",
                      "correct": sum(bool(r["correct"]) for r in rows),
                      "total": len(rows),
                      "regressions_vs_96": regressions(rows, base["probes"]) if base else None,
                      "split_parents": result.get("split_parents"),
                      "split_children": result.get("split_children"),
                      "mean_resident_seats_per_turn": avg("resident"),
                      "mean_wall_ms_per_turn": avg("wall_ms"),
                      "mean_mounted_mass_at_readout": None,
                      "turn_count": len(turns)})
    return table


def evaluate_predictions(table):
    cells = {(r['width'], r['battery']): r for r in table}
    def available(keys):
        return all(k in cells and cells[k]['status'] == 'MEASURED' for k in keys)
    def outcome(keys, predicate):
        return 'NOT_TESTED' if not available(keys) else 'HIT' if predicate() else 'MISS'
    accuracy_keys = [(w, b) for w in (BASELINE, 128, 192) for b in BATTERIES]
    p1 = outcome(accuracy_keys, lambda: all(
        cells[(w, b)]['correct'] >= cells[(BASELINE, b)]['correct']
        for w in (128, 192) for b in BATTERIES))
    p2 = outcome([(64, 'sup'), (BASELINE, 'sup')], lambda:
                 len(cells[(64, 'sup')]['regressions_vs_96']) >= 2)
    latency_keys = [(w, b) for w in (BASELINE, 256) for b in BATTERIES]
    def wall(width):
        rows = [cells[(width, b)] for b in BATTERIES]
        if any(r.get('mean_wall_ms_per_turn') is None or not r.get('turn_count') for r in rows):
            return None
        return sum(r['mean_wall_ms_per_turn'] * r['turn_count'] for r in rows) / sum(r['turn_count'] for r in rows)
    p3 = 'NOT_TESTED'
    if available(latency_keys) and wall(BASELINE) is not None and wall(256) is not None:
        p3 = 'HIT' if wall(256) > wall(BASELINE) and cells[(256, 'census')]['regressions_vs_96'] else 'MISS'
    p4 = outcome([(w, 'longhorizon') for w in WIDTHS if w >= BASELINE], lambda: all(
        cells[(w, 'longhorizon')]['correct'] == cells[(w, 'longhorizon')]['total']
        for w in WIDTHS if w >= BASELINE))
    return [
        {'prediction': '128 and 192 match or beat 96 on all three batteries', 'outcome': p1,
         'scope': 'Accuracy clause; a tie alone does not establish a unique optimum.'},
        {'prediction': '64 loses at least two sup probes vs 96', 'outcome': p2,
         'scope': 'Paired regressions; the failure mechanism requires individual fit receipts.'},
        {'prediction': '256 raises mean wall ms per turn and degrades a census probe', 'outcome': p3,
         'scope': 'Mean wall time weighted by measured turn count across the three batteries.'},
        {'prediction': 'Long-horizon 4/4 at every width >= 96', 'outcome': p4}]


def registration():
    inherited = read(FRAME)
    shape = literal_constants(ROOT / "scripts/grm_eb1_longhorizon_gpu.py",
                              ("GENERATOR_SEED", "TARGET_TURNS", "SHARDS", "SHARD_STOPS"))
    refs = [MAIN / "artifacts/grm_rs3/grm_rs3_part4.json", FRAME, CENSUS]
    sources = ["core/graft_arena.py", "core/graft_repository.py", "core/grm_frame.py",
               "core/grm_demand.py", "core/grm_admission.py",
               "scripts/lsr_p2c_replay_gpu.py", "scripts/lsr_p2c_e2e_gpu.py",
               "scripts/grm_eb1_longhorizon_gpu.py", "scripts/grm_rt1_g2_table.py"]
    logs = sorted((MAIN / "logs").glob("grm_rs4_pytest*.log"))
    return {
        "order": "GRM-WC1", "seat": "wc1-astra",
        "workspace": str(ROOT), "widths": list(WIDTHS), "plan": plan(),
        "registered_before_any_gate": True, "flags": FLAGS,
        "inherited_runtime_flags": inherited["resolved_flags"],
        "generator": shape, "other_seeds": "No new RNG; frozen fixtures and greedy decoding unchanged.",
        "predictions": [
            "96 is NOT the optimum: 128 and 192 match or beat 96 on all three batteries (fewer splits, no co-mount blending because admission is k=1 on these probes).",
            "64 loses at least 2 sup probes (unseatable-after-split class).",
            "256 shows the first sign of the YaRN wall: wall ms per turn rises and at least one census probe degrades; if it does not, say so - the 384 collapse may be a cliff, not a slope.",
            "Long-horizon 4/4 at every width >= 96."],
        "G2": {"required_counts": {"sup": "9/9", "census": "9/10", "longhorizon": "4/4"},
               "semantic_comparator": "lsr_p2c_replay_gpu.reproduction_verdict; expected/rejected value guards retained; exact text also receipted",
               "stop": "A non-reproducing width 96 stops all other widths.",
               "reference_gap": "Main checkout has no grm_rt1 artifacts or rule_on receipts named in grm_rt1_g2_table.py. RS3 Part 4 is 8/9, not RT1's requested 9/9. RT1 reference requested; no substitution claimed."},
        "metrics": {
            "splits": "Count width_guard_parent and width_guard_child metadata in final repositories; sum distinct sup fixtures, never sum cumulative shard snapshots.",
            "residency": "Physical cache rows at end of each served turn; holes are positions, not allocated seats. Report denominators. Sup includes probe turns; census/longhorizon include every session turn.",
            "wall": "Per-turn elapsed wall time excluding model load, lease waits, gaps and session copies; sup includes all ladder work; session uses driver turn_wall_ms.",
            "mounted_mass": "Optional column omitted (null): no extra attention forwards/observer allocations in baseline timing; no unmeasured mass claim."},
        "execution": {"lease": "scripts.grm_cmc1_gpu_arms.gpu_lease on /tmp/forge-gpu.lock; operator right of way",
                      "bounds": "Foreground timeout below ten minutes including lock wait; imported lease/gap bounds, dynamically shortened run allowance after lock acquisition; at least 30 seconds gap inside wrapper.",
                      "process_safety": "No git, no subagents, no external process inspection, no signaling other processes, no lock clearing, no background waiters.",
                      "file_boundary": "New scripts/grm_wc1_*.py, tests, artifacts/grm_wc1 and logs only; native build under artifacts/grm_wc1/native_build.",
                      "model": inherited["model"]},
        "source_records": [record(ROOT / s) for s in sources],
        "reference_records": [record(p) for p in refs],
        "RS4_log_records": [record(p) for p in logs],
        "ambiguities": ["Dispatch fork hash 29c8ef8 differs from order template 29d860b; dispatched path is authoritative; no git used.",
                        "No new numeric constants interpreted as no new experimental tuning constants; inherited runtime, seed, shard and safety limits are preserved."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "plan"))
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(write(OUT / "registration.json", registration(), exclusive=True)))
    else:
        print(json.dumps(plan(), indent=2))


if __name__ == "__main__":
    main()
