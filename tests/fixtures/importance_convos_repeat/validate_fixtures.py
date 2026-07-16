#!/usr/bin/env python3
"""Pure-stdlib validator for the GRM S4 repeat-probe fixture arm."""
import glob
import json
import os
import re
import sys


VALID_NODE_CLASSES = {
    "PROBED_LATER", "STANDING_PREF", "FILLER", "SUPERSEDED", "PROBE",
}
FACT_CLASSES = {"PROBED_LATER", "STANDING_PREF", "SUPERSEDED"}
MIN_TURNS = 28
MAX_TURNS = 40
N_TARGET_FACTS = 4
PROBES_PER_FACT = 2
N_PROBES = N_TARGET_FACTS * PROBES_PER_FACT
MIN_REPEAT_GAP = 8
MIN_CANDIDATES = 6


class FixtureError(Exception):
    pass


def _require(condition, message):
    if not condition:
        raise FixtureError(message)


def _rare_tokens(text):
    out = set()
    for word in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text):
        word = word.rstrip(".,:;")
        if any(ch.isdigit() for ch in word) or (
                word.isupper() and len(word) >= 3):
            out.add(word.lower())
    return out


def validate_turns(data):
    _require(isinstance(data.get("turns"), list), "missing 'turns' list")
    turns = data["turns"]
    _require(MIN_TURNS <= len(turns) <= MAX_TURNS,
             f"turn count {len(turns)} outside [{MIN_TURNS}, {MAX_TURNS}]")
    ids = []
    fact_node_ids = set()
    solo_turns = []
    for pos, turn in enumerate(turns):
        for key in ("turn_id", "role", "text", "node_class", "node_id"):
            _require(key in turn, f"turn missing required key {key!r}: {turn}")
        tid = turn["turn_id"]
        _require(isinstance(tid, int) and not isinstance(tid, bool) and tid > 0,
                 f"turn_id must be a positive int: {tid!r}")
        _require(turn["role"] in ("user", "assistant"),
                 f"turn {tid}: invalid role {turn['role']!r}")
        _require(isinstance(turn["text"], str) and turn["text"].strip(),
                 f"turn {tid}: text must be non-empty")
        node_class = turn["node_class"]
        _require(node_class in VALID_NODE_CLASSES,
                 f"turn {tid}: invalid node_class {node_class!r}")
        node_id = turn["node_id"]
        if node_class in FACT_CLASSES:
            _require(isinstance(node_id, str) and node_id,
                     f"turn {tid}: fact class needs a node_id")
            if node_class != "SUPERSEDED":
                _require(node_id not in fact_node_ids,
                         f"turn {tid}: non-SUPERSEDED node_id reused: {node_id}")
            fact_node_ids.add(node_id)
        else:
            _require(node_id is None,
                     f"turn {tid}: {node_class} must have node_id null")
        ids.append(tid)
        if (turn["role"] == "user" and node_class != "PROBE"
                and (pos + 1 == len(turns)
                     or turns[pos + 1].get("role") == "user")):
            solo_turns.append(tid)
    _require(ids == sorted(ids), "turn_id values are not strictly ascending")
    _require(len(ids) == len(set(ids)), "duplicate turn_id values")
    return turns, solo_turns


def validate_probe_schema(data, turns):
    _require(isinstance(data.get("probes"), list), "missing 'probes' list")
    probes = data["probes"]
    _require(len(probes) == N_PROBES,
             f"expected {N_PROBES} probes, found {len(probes)}")
    by_id = {turn["turn_id"]: turn for turn in turns}
    candidate_sets = []
    for probe in probes:
        for key in ("probe_turn_id", "question", "expected_answer_tokens",
                    "relevance"):
            _require(key in probe,
                     f"probe missing required key {key!r}: {probe}")
        pid = probe["probe_turn_id"]
        _require(pid in by_id, f"probe turn {pid} does not exist")
        turn = by_id[pid]
        _require(turn["role"] == "user" and turn["node_class"] == "PROBE",
                 f"turn {pid} is not a user PROBE")
        _require(probe["question"] == turn["text"],
                 f"probe {pid}: question does not match turn text")
        tokens = probe["expected_answer_tokens"]
        _require(isinstance(tokens, list) and tokens
                 and all(isinstance(t, str) and t for t in tokens),
                 f"probe {pid}: invalid expected_answer_tokens")
        rel = probe["relevance"]
        _require(isinstance(rel, dict) and len(rel) >= MIN_CANDIDATES,
                 f"probe {pid}: need at least {MIN_CANDIDATES} candidates")
        grades = set()
        candidate_ids = set()
        for raw_tid, grade in rel.items():
            try:
                candidate_tid = int(raw_tid)
            except (TypeError, ValueError) as exc:
                raise FixtureError(
                    f"probe {pid}: candidate {raw_tid!r} is not an int") from exc
            _require(candidate_tid in by_id,
                     f"probe {pid}: candidate turn {candidate_tid} is absent")
            _require(candidate_tid < pid,
                     f"probe {pid}: candidate {candidate_tid} is not earlier")
            _require(isinstance(grade, int) and not isinstance(grade, bool)
                     and 0 <= grade <= 3,
                     f"probe {pid}: invalid grade {grade!r} for {candidate_tid}")
            candidate_ids.add(candidate_tid)
            grades.add(grade)
        _require(len(grades) >= 3,
                 f"probe {pid}: grades {sorted(grades)} are not graded")
        top = [tid for tid, grade in rel.items() if grade == 3]
        _require(len(top) == 1,
                 f"probe {pid}: expected one grade-3 target, found {top}")
        candidate_sets.append(candidate_ids)

        pos = next(i for i, item in enumerate(turns) if item["turn_id"] == pid)
        _require(pos + 1 < len(turns),
                 f"probe {pid}: missing scripted assistant answer")
        answer = turns[pos + 1]
        _require(answer["role"] == "assistant"
                 and answer["node_class"] != "PROBE",
                 f"probe {pid}: next turn is not a scripted assistant answer")

    first_candidates = candidate_sets[0]
    _require(all(candidates == first_candidates for candidates in candidate_sets),
             "all probes must grade the same candidate set")
    return probes, first_candidates


def validate_repeat_contract(probes, candidate_ids):
    probes_by_target = {}
    for probe in probes:
        target = int(next(tid for tid, grade in probe["relevance"].items()
                          if grade == 3))
        probes_by_target.setdefault(target, []).append(probe)
    _require(len(probes_by_target) == N_TARGET_FACTS,
             f"expected {N_TARGET_FACTS} target facts, found "
             f"{sorted(probes_by_target)}")
    for target, pair in sorted(probes_by_target.items()):
        _require(len(pair) == PROBES_PER_FACT,
                 f"target turn {target}: expected two probes, found {len(pair)}")
        pair = sorted(pair, key=lambda p: p["probe_turn_id"])
        gap = pair[1]["probe_turn_id"] - pair[0]["probe_turn_id"]
        _require(gap >= MIN_REPEAT_GAP,
                 f"target turn {target}: repeat gap {gap} < {MIN_REPEAT_GAP}")
        _require(pair[0]["expected_answer_tokens"]
                 == pair[1]["expected_answer_tokens"],
                 f"target turn {target}: early/late answer tokens differ")

    controls = candidate_ids - set(probes_by_target)
    _require(len(controls) >= 2,
             f"need at least two never-probed controls, found {sorted(controls)}")
    for control in sorted(controls):
        grades = [probe["relevance"][str(control)] for probe in probes]
        _require(all(0 <= grade <= 2 for grade in grades),
                 f"control turn {control} received a target grade")
        _require(any(grade > 0 for grade in grades),
                 f"control turn {control} is ungraded zero-only filler")
    return sorted(probes_by_target), sorted(controls)


def validate_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("conversation_id", "description", "turns", "probes"):
        _require(key in data, f"missing top-level key {key!r}")
    stem = os.path.splitext(os.path.basename(path))[0]
    _require(stem == data["conversation_id"],
             f"conversation_id {data['conversation_id']!r} != {stem!r}")
    turns, solos = validate_turns(data)
    probes, candidates = validate_probe_schema(data, turns)
    targets, controls = validate_repeat_contract(probes, candidates)
    identifiers = set()
    for turn in turns:
        identifiers |= _rare_tokens(turn["text"])
    return {
        "conversation_id": data["conversation_id"],
        "n_turns": len(turns),
        "n_probes": len(probes),
        "targets": targets,
        "controls": controls,
        "solos": solos,
        "identifiers": identifiers,
    }


def main(argv):
    fixture_dir = (argv[1] if len(argv) > 1
                   else os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(fixture_dir, "convo_*.json")))
    if len(paths) != 4:
        print(f"FAIL expected 4 convo_*.json files, found {len(paths)}")
        return 1
    summaries = []
    failed = 0
    for path in paths:
        try:
            summary = validate_file(path)
        except (FixtureError, json.JSONDecodeError, OSError) as exc:
            print(f"FAIL {os.path.basename(path)}: {exc}")
            failed += 1
            continue
        summaries.append(summary)
        print(f"OK   {os.path.basename(path)}: {summary['n_turns']} turns, "
              f"{summary['n_probes']} probes, targets={summary['targets']}, "
              f"controls={summary['controls']}, solos={summary['solos']}")

    owners = {}
    for summary in summaries:
        for identifier in summary["identifiers"]:
            owners.setdefault(identifier, []).append(summary["conversation_id"])
    collisions = {token: files for token, files in owners.items()
                  if len(set(files)) > 1}
    if collisions:
        print(f"FAIL cross-file identifier collisions: {collisions}")
        failed += 1
    if summaries and not any(summary["solos"] for summary in summaries):
        print("FAIL no SOLO user turns found in the fixture arm")
        failed += 1
    if failed:
        print(f"\n{len(summaries)}/{len(paths)} files valid; failures={failed}")
        return 1
    print(f"\n4/4 files valid, {sum(s['n_probes'] for s in summaries)} "
          "probes total, cross-file identifiers unique, SOLO coverage present")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
