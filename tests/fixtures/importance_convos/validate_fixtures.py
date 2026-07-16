#!/usr/bin/env python3
"""Pure-stdlib validator for the WO-4 labeled conversation fixtures
(GRM importance-weighting program, G1/G2 gates). See README.md for the
full schema writeup — this module is the executable form of that schema.

No model imports, no engine imports, no GPU. Safe to run anywhere.

Usage:
    python3 validate_fixtures.py [dir]

Exits 0 and prints a per-file summary on success; exits 1 and prints the
first violation per file on failure.
"""
import glob
import json
import os
import sys

VALID_NODE_CLASSES = {
    "PROBED_LATER",
    "STANDING_PREF",
    "FILLER",
    "SUPERSEDED",
    "PROBE",
}
# Classes that plant a fact a probe can target (i.e. carry a node_id).
FACT_CLASSES = {"PROBED_LATER", "STANDING_PREF", "SUPERSEDED"}
MIN_TURNS = 15
MAX_TURNS = 25
MIN_PROBES = 3
MIN_CANDIDATES_PER_PROBE = 4
MIN_RELEVANCE_GRADE = 0
MAX_RELEVANCE_GRADE = 3
STANDING_PREF_MIN_UNUSED_GAP = 10


class FixtureError(Exception):
    pass


def _require(cond, msg):
    if not cond:
        raise FixtureError(msg)


def validate_turns(data):
    _require("turns" in data and isinstance(data["turns"], list), "missing 'turns' list")
    turns = data["turns"]
    _require(MIN_TURNS <= len(turns) <= MAX_TURNS,
             f"turn count {len(turns)} outside [{MIN_TURNS}, {MAX_TURNS}]")

    seen_ids = []
    node_ids_seen = set()
    fact_turns = {}  # node_id -> turn_id (first-planted turn)
    for t in turns:
        for key in ("turn_id", "role", "text", "node_class"):
            _require(key in t, f"turn missing required key {key!r}: {t}")
        _require(isinstance(t["turn_id"], int) and t["turn_id"] > 0,
                  f"turn_id must be a positive int: {t.get('turn_id')!r}")
        _require(t["role"] in ("user", "assistant"),
                  f"turn {t['turn_id']}: invalid role {t['role']!r}")
        _require(isinstance(t["text"], str) and t["text"].strip(),
                  f"turn {t['turn_id']}: text must be non-empty string")
        _require(t["node_class"] in VALID_NODE_CLASSES,
                  f"turn {t['turn_id']}: invalid node_class {t['node_class']!r}")
        seen_ids.append(t["turn_id"])

        node_id = t.get("node_id")
        if t["node_class"] in FACT_CLASSES:
            if node_id is not None:
                # SUPERSEDED reuses the same node_id across the original
                # statement and its correction turn(s) by design; other
                # fact classes must not repeat a node_id.
                if node_id in node_ids_seen and t["node_class"] != "SUPERSEDED":
                    raise FixtureError(
                        f"turn {t['turn_id']}: node_id {node_id!r} reused "
                        f"by a non-SUPERSEDED class")
                node_ids_seen.add(node_id)
                fact_turns.setdefault(node_id, t["turn_id"])
        else:
            _require(node_id is None,
                      f"turn {t['turn_id']}: node_class {t['node_class']} "
                      f"must have node_id null, got {node_id!r}")

    _require(seen_ids == sorted(seen_ids),
              "turn_id values must be strictly ascending in file order")
    _require(len(set(seen_ids)) == len(seen_ids), "duplicate turn_id values")
    return turns, fact_turns


def validate_probes(data, turns):
    _require("probes" in data and isinstance(data["probes"], list), "missing 'probes' list")
    probes = data["probes"]
    _require(len(probes) >= MIN_PROBES, f"only {len(probes)} probes, need >= {MIN_PROBES}")

    turn_ids = {t["turn_id"] for t in turns}
    turn_by_id = {t["turn_id"]: t for t in turns}

    for p in probes:
        for key in ("probe_turn_id", "question", "expected_answer_tokens", "relevance"):
            _require(key in p, f"probe missing required key {key!r}: {p}")
        pid = p["probe_turn_id"]
        _require(pid in turn_ids, f"probe_turn_id {pid} does not match any turn_id")
        _require(turn_by_id[pid]["node_class"] == "PROBE",
                  f"turn {pid} referenced as probe_turn_id must have node_class PROBE")
        _require(isinstance(p["question"], str) and p["question"].strip(),
                  f"probe at turn {pid}: question must be non-empty string")
        _require(p["question"] == turn_by_id[pid]["text"],
                  f"probe at turn {pid}: question text must match the turn's text")
        _require(isinstance(p["expected_answer_tokens"], list) and p["expected_answer_tokens"],
                  f"probe at turn {pid}: expected_answer_tokens must be a non-empty list")
        for tok in p["expected_answer_tokens"]:
            _require(isinstance(tok, str) and tok,
                      f"probe at turn {pid}: expected_answer_tokens entries must be non-empty strings")

        rel = p["relevance"]
        _require(isinstance(rel, dict), f"probe at turn {pid}: relevance must be an object")
        _require(len(rel) >= MIN_CANDIDATES_PER_PROBE,
                  f"probe at turn {pid}: only {len(rel)} candidate mounts, "
                  f"need >= {MIN_CANDIDATES_PER_PROBE}")
        grades_present = set()
        for k, v in rel.items():
            try:
                k_int = int(k)
            except (TypeError, ValueError):
                raise FixtureError(f"probe at turn {pid}: relevance key {k!r} not an int-like turn_id")
            _require(k_int in turn_ids, f"probe at turn {pid}: relevance key {k} not a real turn_id")
            _require(k_int < pid, f"probe at turn {pid}: relevance candidate turn {k_int} "
                                    f"must precede the probe turn")
            _require(isinstance(v, int) and MIN_RELEVANCE_GRADE <= v <= MAX_RELEVANCE_GRADE,
                      f"probe at turn {pid}: relevance grade for turn {k} = {v!r} "
                      f"outside [{MIN_RELEVANCE_GRADE}, {MAX_RELEVANCE_GRADE}]")
            grades_present.add(v)
        # G1 needs graded (non-binary) relevance: at least 3 distinct grade
        # values among the candidates, so rank correlation is well-posed.
        _require(len(grades_present) >= 3,
                  f"probe at turn {pid}: relevance grades {sorted(grades_present)} "
                  f"are not graded enough (need >= 3 distinct values across "
                  f"[{MIN_RELEVANCE_GRADE}, {MAX_RELEVANCE_GRADE}])")
    return probes


def validate_standing_pref_isolation(data, turns, probes):
    """G2 discriminator class: a STANDING_PREF fact must accrue ZERO
    references between its planting turn and the probe that targets it,
    with a gap of at least STANDING_PREF_MIN_UNUSED_GAP turns. This is
    what makes a purely retrospective usage signal (S1) fail it by
    construction, per the plan's pre-registered G2 expectation.

    We can only mechanically check the *labeling* half of this contract
    (no other turn shares the STANDING_PREF's node_id, and the gap to
    its probe is wide enough) — whether authored turn *text* leaks a
    reference is a semantic property; the README documents the authoring
    discipline required, and the labels are inspected here as an
    approximation.
    """
    standing = [t for t in turns if t["node_class"] == "STANDING_PREF"]
    for t in standing:
        node_id = t["node_id"]
        _require(node_id, f"STANDING_PREF turn {t['turn_id']} missing node_id")
        plant_turn = t["turn_id"]

        # No other turn (of any class) may carry the same node_id — a
        # second touch would be a "use."
        reuses = [ot["turn_id"] for ot in turns
                  if ot is not t and ot.get("node_id") == node_id]
        _require(not reuses,
                  f"STANDING_PREF {node_id} (turn {plant_turn}) is referenced "
                  f"again at turn(s) {reuses} -- must be zero-use until probed")

        # Must be the top-ranked (or tied-top) candidate in at least one
        # probe's relevance map -- that probe is its "designated" probe,
        # the intended answer rather than an incidental distractor -- and
        # that designated probe must sit outside the unused-gap window.
        # (The node_id may also appear at low grade in OTHER probes as a
        # distractor; that's fine and not checked here.)
        mentions = [p for p in probes if str(plant_turn) in p["relevance"]]
        _require(mentions,
                  f"STANDING_PREF {node_id} (turn {plant_turn}) is never "
                  f"listed in any probe's relevance map")
        designated = [p for p in mentions
                      if p["relevance"][str(plant_turn)] == max(p["relevance"].values())]
        _require(designated,
                  f"STANDING_PREF {node_id} (turn {plant_turn}) is never the "
                  f"top-graded candidate in any probe -- it needs one probe "
                  f"where it is the intended answer")
        gap_ok = [p for p in designated
                  if p["probe_turn_id"] - plant_turn >= STANDING_PREF_MIN_UNUSED_GAP]
        _require(gap_ok,
                  f"STANDING_PREF {node_id}: its designated (top-graded) probe(s) "
                  f"{[p['probe_turn_id'] for p in designated]} are all within "
                  f"{STANDING_PREF_MIN_UNUSED_GAP} turns of planting turn {plant_turn}")


def validate_superseded(data, turns):
    """SUPERSEDED facts must appear at least twice under the same
    node_id: an original statement and >=1 correction. A probe targeting
    that node_id should be answerable by the correction, not the stale
    original -- checked qualitatively in review, but we do mechanically
    require >= 2 touches so the "correction" half of the class is real.
    """
    by_node = {}
    for t in turns:
        if t["node_class"] == "SUPERSEDED":
            by_node.setdefault(t["node_id"], []).append(t["turn_id"])
    for node_id, tids in by_node.items():
        _require(len(tids) >= 2,
                  f"SUPERSEDED {node_id} only touched once at turn(s) {tids}; "
                  f"needs an original statement + >=1 correction")


def validate_class_mix(turns):
    classes_present = {t["node_class"] for t in turns}
    required = {"PROBED_LATER", "STANDING_PREF", "FILLER", "SUPERSEDED", "PROBE"}
    missing = required - classes_present
    _require(not missing, f"missing node classes in this conversation: {sorted(missing)}")


def validate_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("conversation_id", "description", "turns", "probes"):
        _require(key in data, f"file missing required top-level key {key!r}")
    _require(os.path.splitext(os.path.basename(path))[0].endswith(data["conversation_id"])
              or data["conversation_id"] in os.path.basename(path),
              f"conversation_id {data['conversation_id']!r} does not match filename {path!r}")

    turns, _fact_turns = validate_turns(data)
    probes = validate_probes(data, turns)
    validate_class_mix(turns)
    validate_superseded(data, turns)
    validate_standing_pref_isolation(data, turns, probes)

    class_counts = {}
    for t in turns:
        class_counts[t["node_class"]] = class_counts.get(t["node_class"], 0) + 1

    return {
        "conversation_id": data["conversation_id"],
        "n_turns": len(turns),
        "n_probes": len(probes),
        "class_counts": class_counts,
    }


def main(argv):
    fixtures_dir = argv[1] if len(argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    paths = sorted(glob.glob(os.path.join(fixtures_dir, "convo_*.json")))
    if not paths:
        print(f"no convo_*.json files found in {fixtures_dir}", file=sys.stderr)
        return 1

    n_ok = 0
    n_fail = 0
    total_probes = 0
    for path in paths:
        name = os.path.basename(path)
        try:
            summary = validate_file(path)
        except FixtureError as e:
            print(f"FAIL {name}: {e}")
            n_fail += 1
            continue
        except (json.JSONDecodeError, OSError) as e:
            print(f"FAIL {name}: could not load ({e})")
            n_fail += 1
            continue
        n_ok += 1
        total_probes += summary["n_probes"]
        counts = summary["class_counts"]
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"OK   {name}: {summary['n_turns']} turns, "
              f"{summary['n_probes']} probes, classes[{counts_str}]")

    print(f"\n{n_ok}/{len(paths)} files valid, {total_probes} probes total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
