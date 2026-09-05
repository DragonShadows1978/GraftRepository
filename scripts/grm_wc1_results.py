"""Assemble WC1 evidence without turning absent runs or references into passes."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.grm_wc1_sweep import (BASELINE, BATTERIES, CENSUS, MAIN, OUT, WIDTHS,
                                  assemble, evaluate_predictions, read, record, write)


def collect_sup(width):
    from scripts.lsr_p2c_replay_gpu import SUP_SESSIONS
    paths = [OUT / f"width_{width}/sup/{session}.json" for session in SUP_SESSIONS]
    if not all(p.exists() for p in paths):
        return None
    receipts = [read(p) for p in paths]
    probes = [{"probe_id": r["probe_id"], "answer": r["served_answer"],
               "correct": bool(r["verdict"]["correct"]),
               "expected_values": r["expected_values"],
               "rejected_values": r["rejected_values"]}
              for receipt in receipts for r in receipt["probes"]]
    return {"width": width, "battery": "sup", "probes": probes,
            "turns": [t for r in receipts for t in r["wc1_turns"]],
            "split_parents": sum(r["wc1_counts"]["split_parents"] for r in receipts),
            "split_children": sum(r["wc1_counts"]["split_children"] for r in receipts),
            "receipts": [record(p) for p in paths]}


def collect_session(width, battery):
    from scripts import lsr_p2c_replay_gpu as sup
    from scripts import lsr_p2c_e2e_gpu as census
    from scripts import grm_eb1_longhorizon_gpu as lh
    from scripts.grm_det1_common import contains_value
    sup.CENSUS = CENSUS
    module = census if battery == "census" else lh
    paths = [OUT / f"width_{width}/{battery}/{s}.json" for s in module.SHARDS]
    if not all(p.exists() for p in paths):
        return None
    receipts = [read(p) for p in paths]
    session_dir = Path(receipts[-1]["session_dir"])
    scores = {int(p["turn"]): p for p in read(session_dir / "probe_scorecard.json")["probes"]}
    probes = []
    if battery == "census":
        for p in census.e2e_census_probes():
            row = scores[int(p["turn"])]
            answer = row["answer"]
            probes.append({"probe_id": p["probe_id"], "turn": p["turn"], "answer": answer,
                           "correct": any(contains_value(answer, v) for v in p["expected_values"])})
    else:
        for p in lh.extension_plan()["extension_probes"]:
            row = scores[int(p["turn"])]
            probes.append({"probe_id": f"lh_t{p['turn']}_{p['fact_id'].replace(' ', '_')}",
                           "turn": p["turn"], "distance": p["distance"],
                           "answer": row["answer"],
                           "correct": contains_value(row["answer"], p["expected"])})
    instrument = {int(r["turn"]): r for line in (session_dir / "instrumentation.jsonl").read_text().splitlines()
                  if (r := json.loads(line)).get("schema") == "grm_e2e_session_turn_v1"}
    turns = [{**t, "wall_ms": instrument[int(t["turn"])]["turn_wall_ms"]}
             for r in receipts for t in r["wc1_turns"]]
    if len({t['turn'] for t in turns}) != len(turns):
        raise ValueError("duplicate session turns across shards")
    return {"width": width, "battery": battery, "probes": probes, "turns": turns,
            **receipts[-1]["wc1_counts"], "receipts": [record(p) for p in paths]}


def references():
    from scripts.grm_rt1_g2_table import ARMS
    part4 = read(MAIN / "artifacts/grm_rs3/grm_rs3_part4.json")
    result = {"sup": {}, "census": {}, "longhorizon": {}}
    rt1_paths = [MAIN / "artifacts/lsr_p2c" / p for p in ARMS['rule_on']['receipts']]
    missing = [str(p) for p in rt1_paths if not p.exists()]
    if not missing:
        for path in rt1_paths:
            for r in read(path)['probes']:
                result['sup'][r['probe_id']] = r['served_answer']
    else:
        # RS3's correct rows remain comparable; the RT1 changed solace row is
        # explicitly absent, never relabeled as RT1 reproduction.
        result['sup'] = {r['probe_id']: r['served_answer'] for r in part4['G4_sup']['table']
                         if r['rs3_pair_correct']}
    result['census'] = {r['probe_id']: r['served_answer'] for r in part4['G4_census']['table']}
    for key, payload in part4.items():
        if isinstance(payload, dict):
            for r in payload.get('table', []):
                if 'distance' in r and r['distance'] in (36, 44, 52, 60):
                    pid = f"lh_t{r['turn']}_{r['fact'].replace(' ', '_')}"
                    result['longhorizon'][pid] = r['served']
    return result, missing


def reproduction(results):
    from scripts.lsr_p2c_replay_gpu import reproduction_verdict
    refs, missing = references()
    rows = []
    counts = {'sup': (9, 9), 'census': (9, 10), 'longhorizon': (4, 4)}
    indexed = {r['battery']: r for r in results if r['width'] == BASELINE}
    for battery in BATTERIES:
        result = indexed.get(battery)
        required_correct, required_total = counts[battery]
        probes = (result or {}).get('probes', [])
        comparisons = []
        for p in probes:
            reference = refs[battery].get(p['probe_id'])
            comparisons.append({'probe_id': p['probe_id'], 'answer': p['answer'],
                                'reference_answer': reference,
                                'reference_found': reference is not None,
                                'exact_text_equal': p['answer'] == reference if reference is not None else None,
                                'semantic_equal': reproduction_verdict(p['answer'], reference)['reproduced']
                                if reference is not None else None})
        correct = sum(p['correct'] for p in probes)
        count_pass = result is not None and (correct, len(probes)) == counts[battery]
        semantics_pass = bool(probes) and all(p['semantic_equal'] is True for p in comparisons)
        status = ('NOT_RUN' if result is None else 'RED' if not count_pass
                  or any(p['semantic_equal'] is False for p in comparisons)
                  else 'BLOCKED_REFERENCE' if not semantics_pass else 'GREEN')
        rows.append({'battery': battery, 'required': f'{required_correct}/{required_total}',
                     'correct': correct if result else None, 'total': len(probes) if result else None,
                     'status': status, 'count_pass': count_pass,
                     'semantic_matches': sum(p['semantic_equal'] is True for p in comparisons),
                     'semantic_reference_count': sum(p['reference_found'] for p in comparisons),
                     'probes': comparisons})
    status = ('RED' if any(r['status'] == 'RED' for r in rows) else 'GREEN'
              if all(r['status'] == 'GREEN' for r in rows) else 'INCOMPLETE')
    return {'status': status, 'rows': rows, 'missing_rt1_references': missing}


def pytest_summary():
    paths = sorted((ROOT / 'logs').glob('grm_wc1_pytest*.log'))
    result = []
    for path in paths:
        text = path.read_text(errors='replace')
        lines = text.splitlines()
        result.append({'record': record(path),
                       'summary_lines': [l for l in lines if re.search(r'\b(?:passed|failed|skipped|error|errors)\b.* in \d', l)],
                       'failures': [l for l in lines if l.startswith(('FAILED ', 'ERROR '))],
                       'timed_out': 'TimeoutExpired' in text or 'GPU lease exceeded' in text})
    return result


def main():
    results = []
    for width in WIDTHS:
        for battery in BATTERIES:
            result = collect_sup(width) if battery == 'sup' else collect_session(width, battery)
            if result is not None:
                results.append(result)
    write(OUT / 'batteries.json', results)
    gate = reproduction(results)
    errors = [read(p) for p in sorted(OUT.glob('width_96/*/*_error.json'))]
    if errors and gate['status'] == 'INCOMPLETE':
        gate['status'] = 'BLOCKED_EXECUTION'
        gate['execution_errors'] = errors
        for row in gate['rows']:
            if any(e['battery'] == row['battery'] for e in errors) and row['status'] == 'NOT_RUN':
                row['status'] = 'BLOCKED_BEFORE_PROBES'
    write(OUT / 'g2.json', gate)
    table = assemble(results)
    for cell in table:
        if any(e['battery'] == cell['battery'] and e['width'] == cell['width'] for e in errors):
            if cell['status'] == 'NOT_RUN':
                cell['status'] = 'BLOCKED_EXECUTION'
    write(OUT / 'sweep_table.json', table)
    write(OUT / 'predictions.json', evaluate_predictions(table))
    write(OUT / 'pytest_summary.json', pytest_summary())
    print(json.dumps({'G2': gate['status'], 'table': table}, indent=2))


if __name__ == '__main__':
    main()
