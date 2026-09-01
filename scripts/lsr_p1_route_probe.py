#!/usr/bin/env python3
"""LSR Phase 1 — router per-candidate score table (the M6 attribution).

WHAT THIS ANSWERS
-----------------
The lived DET1.5 snapshots persist ``admission.ranking`` (the RAW output of
``ArenaCache.route()``) but not the router's per-candidate score table.  For
two probes the ranking is SHORTER than the eligible universe:

    sup_lumen_head     ranking [1, 0]  over 4 eligible nodes
    sup_orion_current  ranking [0]     over 3 eligible nodes

and the lawful control ``sup_harbor_restatement`` shows the same shape with
``candidate_count=4``, ``production_route_limit=6``, ``raw_ranking_ids=[1,0]``
persisted at the DET1.5 eval stage.  Since
``core/grm_admission.py::decisive_admission_profile`` asks for
``want = min(len(eligible), max(3, route_limit))``, a short ranking cannot be
a window bound.  Something inside ``route()`` dropped candidates.

``ArenaCache.route()`` has exactly THREE places a candidate can vanish
(core/graft_arena.py, the Python backend):

  D1  ``_vector_route_scores`` line 813 -- the vectorized centroid path
      keeps only ``np.isfinite(score)`` entries.  A dropped id never
      enters ``base``.
  D2  the Python per-candidate fallback (lines 748-752) -- same
      ``np.isfinite`` guard on ``_cent_score``, used only when
      ``_vector_route_scores`` returns None.
  D3  the lexical rescore loop (lines 770-775) -- ``if i not in base:
      continue`` re-drops anything D1/D2 already removed, then applies a
      SECOND ``np.isfinite(score)`` guard after adding ``_lex_bonus``.

Collectively these are the "M6 law" drops.  This instrument reproduces the
exact production arithmetic, records EVERY candidate's score at every stage,
and classifies each candidate:

  M6-FILTER-CONVICTED   the candidate was dropped by a non-finite guard
                        (D1/D2/D3); it could never have been ranked.
  SCORE-RANK-CONVICTED  the candidate scored finite and WAS ranked, but
                        below the truncation point / below the served node
                        -- an ordering outcome, not a filter drop.
  OTHER                 finite and ranked within the returned window; not
                        a discard at this stage.

VERDICT SCOPE.  This names the mechanism behind the truncated ranking with
numbers.  It does not re-serve and does not by itself move any Phase 1 stage
verdict: ROUTE-RANK already stands on the persisted ``admission.ranking``.

BOUNDARIES.  Read-only against the repository; writes only content-addressed
artifacts under ``artifacts/lsr_p1/`` with their own provenance (NOT a DET
envelope).  No git, no network, no subagents.  ``selftest`` needs no GPU and
no weights.  ``score-table`` takes a GPU self-lease per probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUTDIR = os.path.join(REPO, "artifacts", "lsr_p1")

CLASS_M6 = "M6-FILTER-CONVICTED"
CLASS_RANK = "SCORE-RANK-CONVICTED"
CLASS_OTHER = "OTHER"

# Drop-site identifiers, named for the production lines they mirror.
D1 = "D1_vector_route_scores_isfinite"        # graft_arena.py:813
D2 = "D2_python_cent_score_isfinite"          # graft_arena.py:748-752
D3_MISSING = "D3_not_in_base"                 # graft_arena.py:771-772
D3_NONFINITE = "D3_lex_rescore_isfinite"      # graft_arena.py:773-775

PROBE_SESSIONS = {
    "sup_lumen_head": ("multi_hop_a_b_c", "What is the current Lumen seal value?"),
    "sup_orion_current": ("short_correction_long_competitor",
                          "What is the current Orion pin value?"),
    "sup_harbor_restatement": ("correction_then_restatement",
                               "What is the current Harbor token value?"),
}

# Lived frame constants, read off the DET1.5 receipts (arena.* fields in the
# snapshot manifests harvested by scripts/lsr_p1_harvest.py).
LIVED_FRAME = {
    "arena_width": 96,
    "topk": 3,
    "max_trips": 1,
    "route_limit": 6,          # max(topk, (max_trips+1)*topk)
    "length_debias": False,    # arena.length_debias in every lived manifest
    "route_backend": "python",  # admission.route_backend in every lived manifest
    "revision_resolution": True,
    "decisive_admission": True,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def classify_score_table(
    *,
    candidates,
    raw_scores,
    lex_bonus,
    route_limit,
    served_ids=(),
    vector_path=True,
    length_debias=False,
    key_lengths=None,
    normalize=True,
):
    """Reproduce ArenaCache.route()'s Python backend and attribute drops.

    Pure function over plain numbers -- no arena, no model, no GPU.  This is
    the piece the selftests exercise, including the non-finite cases.

    ``raw_scores``  : {candidate_id: centroid score} BEFORE debias/normalize.
                      A candidate may be absent entirely (never scored).
    ``lex_bonus``   : {candidate_id: lexical bonus} (production _lex_bonus).
    ``vector_path`` : True mirrors _vector_route_scores (D1); False mirrors
                      the per-candidate _cent_score fallback (D2).

    Returns a dict with the per-candidate table, the resulting ranking, and
    a per-candidate classification.
    """
    candidates = [int(c) for c in candidates]
    key_lengths = dict(key_lengths or {})
    served = {int(v) for v in served_ids}

    drop_site = D1 if vector_path else D2

    # --- stage 1: base scores, with the M6 finite guard (D1 / D2) ---------
    base = {}
    stage1 = {}
    for cid in candidates:
        present = cid in raw_scores
        value = raw_scores.get(cid)
        ok = present and _finite(value)
        stage1[cid] = {
            "scored": bool(present),
            "raw_score": (float(value) if present and _finite(value)
                          else (None if not present else str(value))),
            "finite": bool(ok),
        }
        if ok:
            base[cid] = float(value)

    # --- stage 2: length debias (identity when the flag is off) ----------
    if length_debias and base:
        debiased = {}
        for cid in candidates:
            if cid not in base:
                continue
            score = float(base[cid])
            k = max(1, int(key_lengths.get(cid, 1)))
            norm = float(math.sqrt(math.log2(k + 1.0)))
            debiased[cid] = (score / norm if score >= 0.0 else score * norm)
        base = debiased

    # --- stage 3: normalize (GQA rescale so best |score| sits at 1.0) ----
    if normalize and base:
        mx = max(abs(v) for v in base.values()) + 1e-8
        base = {cid: v / mx for cid, v in base.items()}

    # --- stage 4: lexical rescore + the SECOND finite guard (D3) ---------
    scored = []
    table = []
    for cid in candidates:
        row = {
            "candidate_id": cid,
            "scored_at_all": stage1[cid]["scored"],
            "raw_score": stage1[cid]["raw_score"],
            "raw_finite": stage1[cid]["finite"],
            "normalized_score": (float(base[cid]) if cid in base else None),
            "lex_bonus": (float(lex_bonus.get(cid, 0.0))
                          if _finite(lex_bonus.get(cid, 0.0)) else
                          str(lex_bonus.get(cid))),
        }
        if cid not in base:
            row["final_score"] = None
            row["dropped"] = True
            row["drop_site"] = (
                drop_site if not stage1[cid]["finite"] else D3_MISSING)
            table.append(row)
            continue
        final = base[cid] + float(lex_bonus.get(cid, 0.0)) if _finite(
            lex_bonus.get(cid, 0.0)) else float("nan")
        if not _finite(final):
            row["final_score"] = None
            row["dropped"] = True
            row["drop_site"] = D3_NONFINITE
            table.append(row)
            continue
        row["final_score"] = float(final)
        row["dropped"] = False
        row["drop_site"] = None
        table.append(row)
        scored.append((float(final), cid))

    scored.sort(key=lambda item: -item[0])
    full_ranking = [cid for _, cid in scored]
    ranking = (full_ranking[:int(route_limit)]
               if route_limit is not None else full_ranking)

    # --- classification --------------------------------------------------
    rank_of = {cid: pos for pos, cid in enumerate(full_ranking)}
    returned = set(ranking)
    for row in table:
        cid = row["candidate_id"]
        if row["dropped"]:
            row["classification"] = CLASS_M6
            row["classification_reason"] = (
                "removed by a non-finite guard at %s; it could never appear "
                "in the ranking regardless of order" % row["drop_site"])
            row["full_rank_position"] = None
            continue
        row["full_rank_position"] = rank_of[cid]
        if cid not in returned:
            row["classification"] = CLASS_RANK
            row["classification_reason"] = (
                "scored finite and ranked at position %d, but fell outside "
                "the returned window of %s" % (rank_of[cid], route_limit))
        elif served and cid not in served and any(
                rank_of[cid] > rank_of.get(s, -1) for s in served
                if s in rank_of):
            row["classification"] = CLASS_RANK
            row["classification_reason"] = (
                "scored finite and was returned, but ordered below the "
                "served node(s) %s" % sorted(served))
        else:
            row["classification"] = CLASS_OTHER
            row["classification_reason"] = (
                "finite and returned within the window")

    dropped_ids = [r["candidate_id"] for r in table if r["dropped"]]
    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "route_limit": route_limit,
        "length_debias": bool(length_debias),
        "normalize": bool(normalize),
        "vector_path": bool(vector_path),
        "per_candidate": table,
        "full_ranking": full_ranking,
        "ranking": ranking,
        "ranking_length": len(ranking),
        "ranking_shortfall_vs_candidates": len(candidates) - len(ranking),
        "m6_dropped_ids": dropped_ids,
        "m6_drop_count": len(dropped_ids),
        "shortfall_fully_explained_by_m6": (
            len(candidates) - len(ranking) == len(dropped_ids)),
    }


# ------------------------------------------------------------------ selftest

def _selftest_cases():
    """Synthetic score tables, including the non-finite cases."""
    nan = float("nan")
    inf = float("inf")
    return [
        {
            # Candidate ids are supplied in ASCENDING order while the correct
            # ranking is strictly DESCENDING by score, so an implementation
            # that forgets to sort cannot pass this case.
            "name": "all_finite_no_drops_ranking_is_score_ordered",
            "kwargs": dict(
                candidates=[0, 1, 2, 3],
                raw_scores={0: 0.10, 1: 0.90, 2: 0.50, 3: 0.30},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [1, 2, 3, 0],
            "expect_dropped": [],
            "expect_shortfall": 0,
            "expect_full_rank_position": {1: 0, 2: 1, 3: 2, 0: 3},
        },
        {
            # Exact reversal of candidate order: insertion order [0,1,2] is
            # the exact opposite of the score order [2,1,0].
            "name": "ranking_reverses_candidate_order_when_scores_do",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: 0.10, 1: 0.50, 2: 0.90},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [2, 1, 0],
            "expect_dropped": [],
            "expect_shortfall": 0,
            "expect_full_rank_position": {2: 0, 1: 1, 0: 2},
        },
        {
            "name": "nan_score_drops_two_reproduces_lumen_shape",
            "kwargs": dict(
                candidates=[0, 1, 2, 3],
                raw_scores={0: 0.40, 1: 0.80, 2: nan, 3: nan},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [1, 0],
            "expect_dropped": [2, 3],
            "expect_shortfall": 2,
            # Attribution must name the BASE-score guard, not D3. A NaN raw
            # score that reaches the lexical stage would be mis-attributed,
            # so pin the site explicitly (mutation-tested).
            "expect_drop_site": {2: D1, 3: D1},
        },
        {
            "name": "nan_base_score_never_reaches_D3_attribution",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: 0.5, 1: nan, 2: 0.2},
                lex_bonus={0: 0.0, 1: 1.0, 2: 0.0}, route_limit=6),
            "expect_ranking": [0, 2],
            "expect_dropped": [1],
            "expect_shortfall": 1,
            # Even with a large FINITE lex bonus, a non-finite base score is
            # removed at D1 and can never be rescued by the lexical channel.
            "expect_drop_site": {1: D1},
            "expect_normalized_none": [1],
        },
        {
            "name": "inf_score_is_non_finite_and_drops",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: 0.5, 1: inf, 2: 0.1},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [0, 2],
            "expect_dropped": [1],
            "expect_shortfall": 1,
        },
        {
            "name": "never_scored_candidate_is_a_drop",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: 0.5, 2: 0.1},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [0, 2],
            "expect_dropped": [1],
            "expect_shortfall": 1,
        },
        {
            "name": "nan_lex_bonus_drops_at_D3",
            "kwargs": dict(
                candidates=[0, 1],
                raw_scores={0: 0.5, 1: 0.9},
                lex_bonus={1: nan}, route_limit=6),
            "expect_ranking": [0],
            "expect_dropped": [1],
            "expect_shortfall": 1,
            "expect_drop_site": {1: D3_NONFINITE},
        },
        {
            "name": "orion_shape_one_survivor_of_three",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: 0.7, 1: nan, 2: nan},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [0],
            "expect_dropped": [1, 2],
            "expect_shortfall": 2,
        },
        {
            "name": "window_truncation_is_rank_not_m6",
            "kwargs": dict(
                candidates=[0, 1, 2, 3],
                raw_scores={0: 0.1, 1: 0.9, 2: 0.8, 3: 0.7},
                lex_bonus={}, route_limit=2),
            "expect_ranking": [1, 2],
            "expect_dropped": [],
            "expect_shortfall": 2,
            "expect_class": {3: CLASS_RANK, 0: CLASS_RANK},
        },
        {
            "name": "lex_bonus_dominates_latent_order",
            "kwargs": dict(
                candidates=[0, 1],
                raw_scores={0: 0.10, 1: 0.90},
                lex_bonus={0: 1.0}, route_limit=6),
            "expect_ranking": [0, 1],
            "expect_dropped": [],
            "expect_shortfall": 0,
        },
        {
            # Normalization is NOT a no-op once the lexical channel is live.
            # It rescales the latent channel so the best |score| sits at 1.0,
            # which is exactly what makes the +1 lexical bonus dominant
            # (core/graft_arena.py::_normalize_scores, the GQA override).
            # Raw scores here are small, so WITHOUT normalization the 0.5 lex
            # bonus would decide; WITH it the latent gap is restored to O(1)
            # and candidate 2 wins. A randomized differential check found this
            # changes the ranking in ~19% of mixed lex/latent draws.
            "name": "normalize_rescales_latent_against_lex_channel",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: -0.842, 1: -1.423, 2: -1.529},
                lex_bonus={0: 0.0, 1: 0.5, 2: 0.0}, route_limit=6),
            "expect_ranking": [1, 0, 2],
            "expect_dropped": [],
            "expect_shortfall": 0,
        },
        {
            "name": "normalize_is_rank_preserving_when_lex_is_silent",
            "kwargs": dict(
                candidates=[0, 1, 2],
                raw_scores={0: -0.842, 1: -1.423, 2: -1.529},
                lex_bonus={}, route_limit=6),
            "expect_ranking": [0, 1, 2],
            "expect_dropped": [],
            "expect_shortfall": 0,
        },
        {
            "name": "fallback_path_drop_site_is_D2",
            "kwargs": dict(
                candidates=[0, 1],
                raw_scores={0: 0.5, 1: nan},
                lex_bonus={}, route_limit=6, vector_path=False),
            "expect_ranking": [0],
            "expect_dropped": [1],
            "expect_shortfall": 1,
            "expect_drop_site": {1: D2},
        },
        {
            # SUP-WO1 L1: score / sqrt(log2(K+1)). The LONG node (K=64,
            # normalizer 2.4541) starts AHEAD on raw score and must end up
            # BEHIND the short node (K=1, normalizer 1.0) once debias runs:
            #   no debias -> 0.60 vs 0.90, node 1 wins
            #   debias    -> 0.60 vs 0.367, node 0 wins
            # An implementation that skips debias cannot pass this.
            "name": "length_debias_demotes_the_long_node",
            "kwargs": dict(
                candidates=[0, 1],
                raw_scores={0: 0.60, 1: 0.90},
                lex_bonus={}, route_limit=6,
                length_debias=True, key_lengths={0: 1, 1: 64}),
            "expect_ranking": [0, 1],
            "expect_dropped": [],
            "expect_shortfall": 0,
            "expect_full_rank_position": {0: 0, 1: 1},
        },
        {
            # Same numbers with debias OFF (the lived frame:
            # arena.length_debias == false in every DET1.5 sup manifest).
            # The long node keeps its raw advantage.
            "name": "length_debias_off_keeps_long_node_ahead_lived_frame",
            "kwargs": dict(
                candidates=[0, 1],
                raw_scores={0: 0.60, 1: 0.90},
                lex_bonus={}, route_limit=6,
                length_debias=False, key_lengths={0: 1, 1: 64}),
            "expect_ranking": [1, 0],
            "expect_dropped": [],
            "expect_shortfall": 0,
        },
        {
            "name": "empty_candidate_set",
            "kwargs": dict(
                candidates=[], raw_scores={}, lex_bonus={}, route_limit=6),
            "expect_ranking": [],
            "expect_dropped": [],
            "expect_shortfall": 0,
        },
    ]


def run_selftest():
    results = []
    failures = []
    for case in _selftest_cases():
        got = classify_score_table(**case["kwargs"])
        checks = []

        def check(name, actual, expected):
            ok = actual == expected
            checks.append({"check": name, "ok": ok,
                           "actual": actual, "expected": expected})
            if not ok:
                failures.append("%s: %s actual=%r expected=%r"
                                % (case["name"], name, actual, expected))

        check("ranking", got["ranking"], case["expect_ranking"])
        check("m6_dropped_ids", sorted(got["m6_dropped_ids"]),
              sorted(case["expect_dropped"]))
        check("ranking_shortfall", got["ranking_shortfall_vs_candidates"],
              case["expect_shortfall"])
        for cid, site in (case.get("expect_drop_site") or {}).items():
            row = next(r for r in got["per_candidate"]
                       if r["candidate_id"] == cid)
            check("drop_site[%d]" % cid, row["drop_site"], site)
        for cid, klass in (case.get("expect_class") or {}).items():
            row = next(r for r in got["per_candidate"]
                       if r["candidate_id"] == cid)
            check("classification[%d]" % cid, row["classification"], klass)
        for cid in (case.get("expect_normalized_none") or ()):
            row = next(r for r in got["per_candidate"]
                       if r["candidate_id"] == cid)
            check("normalized_score[%d] is None" % cid,
                  row["normalized_score"], None)
        for cid, pos in (case.get("expect_full_rank_position") or {}).items():
            row = next(r for r in got["per_candidate"]
                       if r["candidate_id"] == cid)
            check("full_rank_position[%d]" % cid, row["full_rank_position"],
                  pos)
        # Invariant: the ranking must be non-increasing in final_score.
        finals = {r["candidate_id"]: r["final_score"]
                  for r in got["per_candidate"] if not r["dropped"]}
        ordered = [finals[c] for c in got["full_ranking"]]
        if ordered != sorted(ordered, reverse=True):
            failures.append("%s: full_ranking is not score-ordered: %r"
                            % (case["name"], ordered))
        # Invariant: every dropped candidate is M6-convicted, and no
        # surviving candidate is.
        for row in got["per_candidate"]:
            if row["dropped"] and row["classification"] != CLASS_M6:
                failures.append("%s: dropped candidate %d not M6-convicted"
                                % (case["name"], row["candidate_id"]))
            if not row["dropped"] and row["classification"] == CLASS_M6:
                failures.append("%s: surviving candidate %d M6-convicted"
                                % (case["name"], row["candidate_id"]))
        results.append({
            "case": case["name"],
            "checks": checks,
            "result": got,
            "passed": all(c["ok"] for c in checks),
        })
    return results, failures


# --------------------------------------------------------------- GPU capture

def _lease(seconds, wait_seconds):
    """Self-lease on the house GPU lock.

    Signature is positional-compatible with
    scripts/grm_cmc1_gpu_arms.py::gpu_lease(seconds, wait_seconds); that
    helper enforces the MAX_LEASE_SECONDS ceiling itself.
    """
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    return gpu_lease(int(seconds), int(wait_seconds))


def capture_score_table(probe, lease_seconds, lock_wait_seconds):
    """Rebuild the lived arena for one probe and capture the real table.

    Uses the lived campaign's OWN loader and node installer
    (scripts/grm_det1_2_gpu.py::_load_model_repo and
    scripts/grm_det1_3_gpu.py::_install_lived_nodes) rather than a
    re-implementation, so the arena scored here is the arena that served,
    with the same M5 edges. The scores are then read through the production
    helpers and re-run through classify_score_table(); the
    ``reconstruction_matches_production`` field is the guard that the
    reconstruction is faithful.
    """
    session, question = PROBE_SESSIONS[probe]
    fixture_path = os.path.join(
        REPO, "tests", "fixtures", "supersession_battery", session + ".json")
    with open(fixture_path, "r", encoding="utf-8") as fh:
        fixture = json.load(fh)

    import pathlib
    import shutil
    import tempfile

    # The lived campaign's own loader. It pins the whole frame that the
    # DET1.5 receipts record (arena_width=96, topk=3, live_turns=2,
    # GptOssGQAArenaCache, revision_resolution=True), so the arena this
    # instrument scores is the arena that served.
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes

    runtime_frame = {"resolved_flags": {
        "adm_decisive": LIVED_FRAME["decisive_admission"]}}

    with _lease(lease_seconds, lock_wait_seconds):
        workdir = tempfile.mkdtemp(prefix="lsr_p1_route_probe_")
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            pathlib.Path(workdir), runtime_frame)
        try:
            arena = repo.arena
            _install_lived_nodes(repo, e2e, fixture)
            frame_check = {
                "arena_width": int(arena.width),
                "topk": int(arena.topk),
                "revision_resolution": bool(
                    getattr(arena, "revision_resolution", False)),
                "decisive_admission": bool(
                    getattr(arena, "decisive_admission", False)),
                "length_debias": bool(getattr(arena, "length_debias", False)),
                "graft_count": len(arena.grafts),
            }
            eligible = [int(i) for i in arena._route_cand_base()]
            probe_key = arena._probe_key(question)
            vector = arena._vector_route_scores(probe_key, eligible)
            vector_path = vector is not None
            raw_scores = {}
            if vector_path:
                raw_scores = {int(k): float(v) for k, v in vector.items()}
            else:
                for i in eligible:
                    raw_scores[int(i)] = float(
                        arena._cent_score(probe_key, arena.grafts[i]))
            qlex = arena._query_lex_tokens(question)
            lex_bonus = {int(i): float(arena._lex_bonus(qlex, arena.grafts[i]))
                         for i in eligible}
            key_lengths = {int(i): int(arena._route_key_length(arena.grafts[i]))
                           for i in eligible}
            # The production call, with the same probe_key, so the
            # reconstruction below is compared against the real router.
            production_ranking = [int(v) for v in (arena.route(
                question, exclude=set(), limit=LIVED_FRAME["route_limit"],
                probe_key=probe_key) or [])]
            backend = str(getattr(arena, "last_route_backend", "unknown"))
            debias = bool(getattr(arena, "length_debias", False))
        finally:
            try:
                repo.close()
            except BaseException:
                pass
            shutil.rmtree(workdir, ignore_errors=True)

    table = classify_score_table(
        candidates=eligible,
        raw_scores=raw_scores,
        lex_bonus=lex_bonus,
        route_limit=LIVED_FRAME["route_limit"],
        vector_path=vector_path,
        length_debias=debias,
        key_lengths=key_lengths,
    )
    table["production_ranking"] = production_ranking
    table["reconstruction_matches_production"] = (
        table["ranking"] == production_ranking)
    table["route_backend"] = backend
    return {
        "probe_id": probe,
        "session_id": session,
        "question": question,
        "fixture": {"path": os.path.relpath(fixture_path, REPO),
                    "sha256": sha256_file(fixture_path)},
        "lived_frame": dict(LIVED_FRAME),
        "frame_as_built": frame_check,
        "frame_matches_lived": (
            frame_check["arena_width"] == LIVED_FRAME["arena_width"]
            and frame_check["topk"] == LIVED_FRAME["topk"]
            and frame_check["revision_resolution"]
            == LIVED_FRAME["revision_resolution"]
            and frame_check["length_debias"] == LIVED_FRAME["length_debias"]),
        "score_table": table,
    }


def write_artifact(kind, payload):
    payload = dict(payload)
    payload.update({
        "schema": "grm.lsr_p1.route_score_table.v1",
        "program": "LSR",
        "phase": "1",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. Reproduces ArenaCache.route()'s Python backend "
            "and attributes every candidate drop to its production line. "
            "Not a DET envelope."),
        "instrument": {
            "path": "scripts/lsr_p1_route_probe.py",
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
        "classification_vocabulary": [CLASS_M6, CLASS_RANK, CLASS_OTHER],
        "drop_sites": {
            D1: "core/graft_arena.py:813 (_vector_route_scores isfinite)",
            D2: "core/graft_arena.py:748-752 (_cent_score isfinite)",
            D3_MISSING: "core/graft_arena.py:771-772 (i not in base)",
            D3_NONFINITE: "core/graft_arena.py:773-775 (lex rescore isfinite)",
        },
    })
    body = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "lsr_p1_%s_%s.json" % (kind, digest[:16]))
    if os.path.exists(out):
        print("EXISTS (append-only, identical content):", out)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("WROTE:", out)
    return out, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("selftest", help="CPU selftests, no GPU/weights")
    p_self.add_argument("--emit", action="store_true",
                        help="write a content-addressed receipt")

    p_score = sub.add_parser("score-table", help="capture one probe (GPU)")
    p_score.add_argument("--probe", required=True,
                         choices=sorted(PROBE_SESSIONS))
    p_score.add_argument("--lease-seconds", type=int, default=580)
    p_score.add_argument("--lock-wait-seconds", type=int, default=7200)

    args = parser.parse_args(argv)

    if args.command == "selftest":
        results, failures = run_selftest()
        for row in results:
            print("%-46s %s" % (row["case"], "PASS" if row["passed"] else "FAIL"))
        print("cases=%d failures=%d" % (len(results), len(failures)))
        for line in failures:
            print("  FAIL:", line)
        if args.emit:
            write_artifact("route_probe_selftest", {
                "status": "PASS" if not failures else "FAIL",
                "case_count": len(results),
                "failure_count": len(failures),
                "failures": failures,
                "cases": results,
            })
        return 1 if failures else 0

    payload = capture_score_table(
        args.probe, args.lease_seconds, args.lock_wait_seconds)
    table = payload["score_table"]
    write_artifact("route_score_table", payload)
    print("probe=%s candidates=%d ranking=%s shortfall=%d m6_dropped=%s"
          % (args.probe, table["candidate_count"], table["ranking"],
             table["ranking_shortfall_vs_candidates"], table["m6_dropped_ids"]))
    print("reconstruction_matches_production=%s"
          % table["reconstruction_matches_production"])
    if not table["reconstruction_matches_production"]:
        print("RED: reconstruction disagrees with production route(); "
              "the attribution below is NOT trustworthy.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
