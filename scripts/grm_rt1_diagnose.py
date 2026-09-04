"""GRM-RT1 Mission 1 — diagnose the solace ranking from receipts, on CPU.

Decomposes the ranking the RS1/RS3 receipts recorded (``[2, 4, 3, 1, 0]`` on
``sup_solace_fresh``) into the two channels ``ArenaCache.route`` actually
adds:

    score(i) = normalized_latent(i) + _lex_bonus(qlex, grafts[i])

The LEXICAL channel and the identifier-binding predicate are pure text laws
and are computed here EXACTLY as production computes them (the frozen ADM1
``is_identifier_binding`` and the arena's own ``_lex_bonus`` /
``_query_lex_tokens`` / ``_rare_tokens``).  The LATENT channel needs the
model, so it is not recomputed: it is taken from the receipts (the ranking
and ``admission_route_margin_1_2``) and reported as measured, never guessed.

The split family's texts are reconstructed from the registered fixture text
of ``sable_competitor`` by the SAME width-guard chunker production used, so
the children's own texts are the ones the router actually scored.

No GPU. No repository write. Read-only over the receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import ArenaCache  # noqa: E402
from core.grm_admission import (  # noqa: E402
    is_identifier_binding,
    normalized_words,
)

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rt1"

#: The main checkout holds the (gitignored) RS1/RS3 receipts.
DEFAULT_RECEIPTS = Path("/mnt/ForgeRealm/GraftRepository")


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_record(path: Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def lex_bonus(qlex: set[str], text: str) -> float:
    """``ArenaCache._lex_bonus`` for a node whose rare set is its own text's.

    Reproduces the production expression exactly: the stored rare-key set
    first, then the node-text residual for query content words that cannot
    hit a rare key.
    """
    if not qlex:
        return 0.0
    have = set(ArenaCache._rare_tokens(text))
    if not (qlex <= have):
        have |= ArenaCache._node_text_tokens(text)
    return len(qlex & have) / len(qlex)


#: The tokenizer the lived run used.  Loaded WEIGHTS-FREE (tokenizer only),
#: so this whole diagnosis stays on CPU and takes no GPU lease.
MODEL_DIR = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


def split_texts(text: str, budget: int) -> tuple[list[str], int]:
    """Reproduce the librarian's width-guard split of one node, on CPU.

    Runs the REAL production path — ``_width_fitting_chunks`` then
    ``_chunk_token_spans``, decoded back through the model tokenizer — so the
    children's texts are the ones the router actually scored, not an
    approximation.  Returns ``(child_texts, parent_ntok)``.
    """
    from transformers import AutoTokenizer

    from core.graft_repository import GraftRepository

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    def encode(value: str) -> list[int]:
        return tokenizer.encode(str(value), add_special_tokens=False)

    arena = ArenaCache.__new__(ArenaCache)
    arena.encode = encode
    arena.decode = tokenizer.decode
    repo = GraftRepository.__new__(GraftRepository)
    repo.arena = arena

    ids = encode(text)
    ntok = len(ids)
    chunks = repo._width_fitting_chunks(text, int(budget), boundary="section")
    spans = repo._chunk_token_spans(chunks, ntok, int(budget))
    if not spans:
        return [text], ntok
    return [tokenizer.decode(ids[a:b]) for a, b in spans], ntok


def identifier_sequence(question: str) -> tuple[list[str], set[str], set[str]]:
    """The production identifier sequence for a question, tokenizer-free.

    Mirrors ``grm_admission.ordered_identifier_tokens`` with the arena's
    static token laws (``_rare_tokens`` / ``_query_lex_tokens``), which are
    ``staticmethod``/``classmethod`` and need no arena instance.
    """
    qrare = {str(v).casefold() for v in ArenaCache._rare_tokens(question)}
    qlex = ArenaCache._query_lex_tokens(question)
    selected = qrare or {str(v).casefold() for v in qlex}
    ordered: list[str] = []
    seen: set[str] = set()
    for word in normalized_words(question):
        if word in selected and word not in seen:
            ordered.append(word)
            seen.add(word)
    ordered.extend(sorted(selected - seen))
    return ordered, qrare, set(qlex)


def diagnose(receipts_root: Path) -> dict[str, Any]:
    rs3_registration = receipts_root / "artifacts/grm_rs3/registration.json"
    rs3_reg = _read(rs3_registration)
    candidates = sorted(
        (receipts_root / "artifacts/grm_rs1").glob(
            "grm_rs1_A0_fresh_fact_controls_*.json"))
    if not candidates:
        raise SystemExit("RS1 A0 fresh_fact_controls receipt not found")
    rs1_a0 = candidates[0]
    a0 = _read(rs1_a0)

    probes = rs3_reg["probes_REGISTERED_BEFORE_ANY_GATE"]
    solace = probes["sup_solace_fresh"]

    fixture_path = (
        ROOT / "tests/fixtures/supersession_battery/fresh_fact_controls.json")
    fixture = _read(fixture_path)
    fixture_text = {n["node_id"]: n["text"] for n in fixture["nodes"]}

    # Node texts by graft index.  The DEPOSITED text is the registered
    # ``fed_text`` (harmony-wrapped), not the bare fixture string, and it is
    # the text the width guard actually chunked — so the split is reproduced
    # from it and the fixture text only fills nodes the registration omits.
    node_text: dict[int, str] = {}
    for node_id, index in solace["fixture_node_to_idx"].items():
        node_text[int(index)] = fixture_text[node_id]
    for entry in solace["live_nodes"]:
        node_text[int(entry["graft_id"])] = str(entry["fed_text"])

    arena_width = int(a0["arena_width"])
    by_probe = {p["probe_id"]: p for p in a0["probes"]}
    solace_info = by_probe["sup_solace_fresh"]["info"]
    children = [int(v) for v in solace_info["fit_split_children"]]
    parent = int(solace_info["fit_split_parent"])

    chunks, parent_ntok = split_texts(node_text[parent], arena_width)
    if len(chunks) != len(children):
        raise SystemExit(
            f"reconstructed {len(chunks)} chunks but the receipt names "
            f"{len(children)} children {children}: the split could not be "
            "reproduced, so the diagnosis would be a guess")
    child_text = {
        child: chunks[position] for position, child in enumerate(children)
    }

    questions = {
        "sup_solace_fresh": fixture["probes"][1]["question"],
        "sup_reserve_tundra_ledger":
            "What is the current Tundra ledger value?",
    }

    rows = []
    for probe_id, question in questions.items():
        info = by_probe[probe_id]["info"]
        ordered, qrare, qlex = identifier_sequence(question)
        ranking = [int(v) for v in info["ranking_ids"]]
        table = []
        for position, index in enumerate(ranking):
            text = child_text.get(index, node_text.get(index, ""))
            is_child = index in set(children)
            is_parent = index == parent
            own_binds = is_identifier_binding(
                candidate_text=text,
                ordered_identifier_tokens=ordered,
                rare_identifier_tokens=qrare,
            )
            # The INHERITED surface: the width guard gives the index parent
            # the UNION of its own rare tokens and every child's, and gives
            # it ``child_cents`` so ``_cent_score`` scores it as the max over
            # the family.  This column is what that union would bind.
            union_text = text
            if is_parent:
                union_text = " ".join(
                    [text] + [child_text.get(c, "") for c in children])
            union_binds = is_identifier_binding(
                candidate_text=union_text,
                ordered_identifier_tokens=ordered,
                rare_identifier_tokens=qrare,
            )
            hits = sorted(
                set(qlex) & (set(ArenaCache._rare_tokens(text))
                             | ArenaCache._node_text_tokens(text)))
            table.append({
                "rank": position + 1,
                "graft_id": index,
                "role": (
                    "split_index_parent" if is_parent
                    else "split_child" if is_child
                    else "fact_node"),
                "lex_bonus": round(lex_bonus(qlex, text), 6),
                "identifier_hits_own_text": hits,
                "binds_own_text": bool(own_binds),
                "binds_union_surface": bool(union_binds),
                "text_head": text[:110],
            })
        rows.append({
            "probe_id": probe_id,
            "question": question,
            "query_rare_tokens": sorted(qrare),
            "query_lex_tokens": sorted(qlex),
            "ordered_identifier_tokens": ordered,
            "ranking_ids": ranking,
            "admission_rank_plan": [
                int(v) for v in info["admission_rank_plan"]],
            "admission_identified_candidates": [
                int(v) for v in info["admission_identified_candidates"]],
            "admission_policy_branch": str(info["admission_policy_branch"]),
            "route_margin_1_2": float(info["admission_route_margin_1_2"]),
            "fit_planned": [int(v) for v in info["fit_planned"]],
            "fit_seated": [int(v) for v in info["fit_seated"]],
            "served_answer": by_probe[probe_id]["served_answer"],
            "correct": bool(by_probe[probe_id]["correct"]),
            "table": table,
        })

    return {
        "schema": "grm.rt1.diagnosis.v1",
        "program": "GRM",
        "phase": "RT1",
        "order": "orders/GRM_RT1_SPLIT_CHILD_ROUTING.md",
        "arena_width": arena_width,
        "latent_channel_note": (
            "The latent term needs the model and is NOT recomputed here; the "
            "ranking and route_margin_1_2 are taken verbatim from the RS1 A0 "
            "receipt.  Only the lexical/identifier channels are recomputed, "
            "with production's own frozen predicates."),
        "split_family": {
            "index_parent": parent,
            "index_parent_ntok": parent_ntok,
            "children": children,
            "child_texts": child_text,
            "child_ntok": {
                child: len(text.split()) for child, text in child_text.items()},
            "chunker": (
                "GraftRepository._width_fitting_chunks + _chunk_token_spans, "
                "decoded through the lived run's own tokenizer"),
            "inheritance": (
                "graft_repository._guard_deposit_width gives the parent the "
                "UNION rare surface and child_cents, so _cent_score scores it "
                "as max(own, children) — the parent and both children all sit "
                "in the routing surface as first-class candidates."),
        },
        "sources": {
            "rs1_a0": _file_record(rs1_a0),
            "rs3_registration": _file_record(rs3_registration),
            "fixture": _file_record(fixture_path),
        },
        "probes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts-root", type=Path,
                        default=DEFAULT_RECEIPTS)
    parser.add_argument("--out", type=Path,
                        default=ARTIFACT_DIR / "grm_rt1_diagnosis.json")
    args = parser.parse_args()
    out = diagnose(args.receipts_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    for probe in out["probes"]:
        print(f"\n=== {probe['probe_id']}  correct={probe['correct']}  "
              f"branch={probe['admission_policy_branch']}")
        print(f"  ranking={probe['ranking_ids']}  "
              f"plan={probe['admission_rank_plan']}  "
              f"identified={probe['admission_identified_candidates']}  "
              f"margin_1_2={probe['route_margin_1_2']}")
        print(f"  qlex={probe['query_lex_tokens']}  "
              f"qrare={probe['query_rare_tokens']}")
        print("  rank gid role                 lex    own    union  text")
        for row in probe["table"]:
            print(f"  {row['rank']:>4} {row['graft_id']:>3} "
                  f"{row['role']:<20} {row['lex_bonus']:.3f}  "
                  f"{str(row['binds_own_text']):<5}  "
                  f"{str(row['binds_union_surface']):<5}  "
                  f"{row['text_head'][:58]!r}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
