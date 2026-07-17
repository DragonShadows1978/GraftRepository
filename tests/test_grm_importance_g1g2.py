"""GRM-IMPORTANCE WO-5 (docs/GRM_IMPORTANCE_PLAN.md, docs/GRM_IMPORTANCE_
LEDGER.md are law) — G1/G2 gate driver + analysis.

Three tiers:

  1. DRIVER (GPU, gated behind `--run-gpu --convo N`, one fixture
     conversation per invocation): loads MiniCPM3, drives a persistent
     ArenaCache-backed GraftRepository through
     tests/fixtures/importance_convos/convo_0N_*.json turn by turn.
     Scripted (non-PROBE) turns are deposited as paired exchanges via
     GraftRepository.add_turn(), or as natural ``User: ...\n`` ArenaCache
     feeds when no scripted assistant half exists,
     followed by repo.idle() so the S2 idle-window salience pass fires
     immediately at deposit time (WO-3, core/graft_repository.py). At each
     PROBE turn: mount EXACTLY the probe's graded candidate set (relevance
     map keys) via ArenaCache._attempt()'s explicit `picks` argument —
     bypassing route()'s top-k entirely so all three signals score the
     identical candidate set — generate the reply greedily with S1
     telemetry on (WO-1, ArenaCache.set_telemetry()/s1_mass()), then
     teacher-force that SAME generated reply for the S3 counterfactual
     sweep (WO-2, tests/test_grm_importance_counterfactual.py: full-set
     reference warmed, then minus-one per candidate). Writes one JSON
     artifact per convo under tests/fixtures/importance_convos/artifacts/
     (schema grm_importance_g1g2_convo_v2; sealed v1 artifacts remain
     readable by the CPU analysis loader).

  2. ANALYSIS (--analyze, CPU only, reads the artifacts written by the
     driver): registered G1 metrics (median Spearman vs S3 ranks per
     signal, top-1 agreement with the load-bearing-winner eligibility
     rule) and G2 metrics (median s2_salience STANDING_PREF vs FILLER;
     s1_mass STANDING_PREF vs FILLER reported, no gate per plan). Prints
     PASS/FAIL per registered threshold plus a JSON summary line (schema
     grm_importance_g1g2_verdict_v1). Pure numpy — no scipy anywhere in
     this repo (checked), so Spearman is hand-rolled here.

  3. CPU UNIT TESTS (pytest, default collection, no GPU, no model load):
     Spearman correctness (incl. ties), top-1 eligibility rule, G1/G2
     threshold logic on synthetic artifacts (pass and fail cases), and
     artifact schema round-trip.

Registered numbers this file consumes verbatim (ledger, "G1/G2 THRESHOLDS
REGISTERED"):
  - noise_floor mean|Δlogit| = 1.881 (G0b) -> eligibility cutoff = 2x
    floor = 3.762. A probe's top S3 dependence below this cutoff means no
    load-bearing winner exists there; that probe's top-1 slot is EXCLUDED
    from every signal's top-1 agreement denominator (not just S1's — the
    plan singles out S1 for the retrospective-failure discussion but the
    eligibility rule itself is about the PROBE, stated once, applying to
    "a signal" generically. Flagged as a spec reading below in the
    driver/report, not silently resolved).
  - G1 PASS = median Spearman >= 0.5 AND top-1 agreement (over eligible
    probes only) >= 50%.
  - G2 PASS (S2 only) = median s2_salience(STANDING_PREF) >= 2 AND
    (median STANDING_PREF - median FILLER) >= 1 rubric point. S1 on
    STANDING_PREF vs FILLER is reported, no pass/fail attached.

S3 metric functions (mean_abs_delta_logit, mean_kl, dependence,
minus_node_picks, all_minus_one_picks) are IMPORTED from
tests/test_grm_importance_counterfactual.py, never duplicated (WO-2 owns
that file).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_grm_importance_counterfactual import (
    dependence, minus_node_picks,
)

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "importance_convos")
ARTIFACTS_DIR = os.path.join(FIXTURES_DIR, "artifacts")

CONVO_ARTIFACT_SCHEMA_V1 = "grm_importance_g1g2_convo_v1"
CONVO_ARTIFACT_SCHEMA_V2 = "grm_importance_g1g2_convo_v2"
# New driver receipts are v2.  Keep the original name as the writer's
# current schema so existing callers of make_convo_artifact() move forward
# without an API change.
CONVO_ARTIFACT_SCHEMA = CONVO_ARTIFACT_SCHEMA_V2
SUPPORTED_CONVO_ARTIFACT_SCHEMAS = frozenset({
    CONVO_ARTIFACT_SCHEMA_V1,
    CONVO_ARTIFACT_SCHEMA_V2,
})
VERDICT_SCHEMA = "grm_importance_g1g2_verdict_v1"

# Registered floors/thresholds (ledger, "G1/G2 THRESHOLDS REGISTERED",
# 2026-07-16, from the G0b gate). Never adjusted after seeing G1/G2 results
# per the house "thresholds registered before the governed gate" law.
NOISE_FLOOR_MEAN_ABS_DLOGIT = 1.881
ELIGIBILITY_CUTOFF = 2.0 * NOISE_FLOOR_MEAN_ABS_DLOGIT   # 3.762
G1_MIN_MEDIAN_SPEARMAN = 0.5
G1_MIN_TOP1_AGREEMENT = 0.5
G2_MIN_MEDIAN_STANDING_PREF = 2.0
G2_MIN_STANDING_PREF_MINUS_FILLER = 1.0

SIGNALS = ("s1_mass", "s2_salience", "s3_dep_dlogit")


# ============================================================================
# Pure-numpy Spearman (no scipy in this repo — checked: `grep -rn "import
# scipy" .` returns nothing outside third-party vendor trees).
# ============================================================================

def _rank_average_ties(values):
    """Average (fractional) ranks, ties get the mean of the ranks they'd
    span — the standard Spearman tie-handling convention (equivalent to
    scipy.stats.rankdata(method='average')). 0-indexed values in, 1-indexed
    average ranks out (rank 1 = smallest)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # positions i..j (0-indexed) tie; average of ranks (i+1)..(j+1)
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation between equal-length sequences a, b.
    Pure numpy, tie-aware (average-rank method). Returns nan if either
    side has zero variance in rank (all values tied — correlation
    undefined), matching scipy's convention of returning nan there."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {a.shape}")
    n = a.shape[0]
    if n < 2:
        raise ValueError(f"need >= 2 points, got {n}")
    ra = _rank_average_ties(a)
    rb = _rank_average_ties(b)
    sa = ra.std()
    sb = rb.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    cov = float(np.mean((ra - ra.mean()) * (rb - rb.mean())))
    return float(cov / (sa * sb))


# ============================================================================
# Top-1 eligibility + agreement
# ============================================================================

def probe_is_eligible(s3_deps_by_candidate):
    """A probe counts toward top-1 agreement only if a load-bearing winner
    exists: the probe's TOP S3 dependence (mean|Δlogit|) must be >= the
    registered eligibility cutoff (2x the G0b noise floor, 3.762).
    s3_deps_by_candidate: {candidate_id: mean_abs_dlogit}."""
    if not s3_deps_by_candidate:
        return False
    return max(s3_deps_by_candidate.values()) >= ELIGIBILITY_CUTOFF


def top1_agreement(signal_scores, s3_deps_by_candidate, candidates):
    """1 if the candidate with the max signal score is ALSO the candidate
    with the max S3 dependence, else 0. Ties on the signal side: if
    multiple candidates share the max signal score AND one of them is the
    S3 top-1, count as agreement (a tie that includes the right answer is
    not a miss). `candidates` fixes iteration order so tie-breaking is
    deterministic and independent of dict ordering quirks."""
    s3_top = max(candidates, key=lambda c: s3_deps_by_candidate[c])
    s3_top_val = s3_deps_by_candidate[s3_top]
    s3_winners = {c for c in candidates if s3_deps_by_candidate[c] == s3_top_val}
    sig_top_val = max(signal_scores[c] for c in candidates)
    sig_winners = {c for c in candidates if signal_scores[c] == sig_top_val}
    return 1 if (sig_winners & s3_winners) else 0


# ============================================================================
# G1/G2 threshold logic — pure data in, verdict dict out. Exercised directly
# by the CPU unit tests on synthetic artifacts; the --analyze CLI path is a
# thin wrapper that loads real artifacts and calls this.
# ============================================================================

def compute_g1(records):
    """records: list of per-probe dicts, each:
        {"probe_id": ..., "candidates": [{"candidate_id", "ground_truth_grade",
         "s1_mass", "s2_salience", "s3_dep_dlogit", "s3_dep_kl"}, ...]}
    Returns {signal: {"median_spearman", "top1_agreement", "n_eligible",
    "n_probes", "pass"}} for s1_mass and s2_salience (S1/S2 are the
    candidate signals gated against S3 per the plan; S3 is the arbiter,
    never gated against itself)."""
    out = {}
    for signal in ("s1_mass", "s2_salience"):
        spearmans = []
        agreements = []
        n_eligible = 0
        for rec in records:
            cands = rec["candidates"]
            ids = [c["candidate_id"] for c in cands]
            gt = [c["ground_truth_grade"] for c in cands]
            sig = {c["candidate_id"]: c[signal] for c in cands}
            s3 = {c["candidate_id"]: c["s3_dep_dlogit"] for c in cands}
            sig_vals = [sig[i] for i in ids]
            if any(v is None for v in sig_vals):
                # a signal value is missing for this probe (e.g. S2 parse
                # failure -> None per the QC "never a guess" law) — this
                # probe cannot be ranked for this signal at all; excluded
                # from BOTH the Spearman list and the eligibility/top-1
                # count for this signal, not silently zero-filled.
                continue
            sp = spearman(gt, sig_vals)
            if not (sp != sp):   # not NaN
                spearmans.append(sp)
            if probe_is_eligible(s3):
                n_eligible += 1
                agreements.append(top1_agreement(sig, s3, ids))
        median_sp = float(np.median(spearmans)) if spearmans else float("nan")
        top1 = (sum(agreements) / len(agreements)) if agreements else float("nan")
        gate_pass = bool(
            spearmans and agreements
            and median_sp >= G1_MIN_MEDIAN_SPEARMAN
            and top1 >= G1_MIN_TOP1_AGREEMENT)
        out[signal] = {
            "median_spearman": median_sp,
            "top1_agreement": top1,
            "n_probes_ranked": len(spearmans),
            "n_eligible": n_eligible,
            "n_probes_total": len(records),
            "pass": gate_pass,
        }
    return out


def compute_g2(standing_pref_records, filler_records):
    """standing_pref_records / filler_records: lists of per-candidate dicts
    (the STANDING_PREF-class and FILLER-class candidates pooled across all
    probes/convos), each with at least "s2_salience" and "s1_mass".
    Returns the registered G2 verdict (S2 gated; S1 reported only)."""
    def _vals(records, key):
        return [r[key] for r in records if r.get(key) is not None]

    s2_pref = _vals(standing_pref_records, "s2_salience")
    s2_filler = _vals(filler_records, "s2_salience")
    s1_pref = _vals(standing_pref_records, "s1_mass")
    s1_filler = _vals(filler_records, "s1_mass")

    median_s2_pref = float(np.median(s2_pref)) if s2_pref else float("nan")
    median_s2_filler = float(np.median(s2_filler)) if s2_filler else float("nan")
    delta = (median_s2_pref - median_s2_filler
             if s2_pref and s2_filler else float("nan"))
    gate_pass = bool(
        s2_pref and s2_filler
        and median_s2_pref >= G2_MIN_MEDIAN_STANDING_PREF
        and delta >= G2_MIN_STANDING_PREF_MINUS_FILLER)

    return {
        "s2_salience": {
            "median_standing_pref": median_s2_pref,
            "median_filler": median_s2_filler,
            "delta": delta,
            "n_standing_pref": len(s2_pref),
            "n_filler": len(s2_filler),
            "pass": gate_pass,
        },
        # S1 on STANDING_PREF is a REGISTERED EXPECTATION of failure
        # (retrospective signal, zero uses by construction) — reported,
        # no pass/fail attached (plan + ledger, both explicit on this).
        "s1_mass": {
            "median_standing_pref": (float(np.median(s1_pref))
                                     if s1_pref else float("nan")),
            "median_filler": (float(np.median(s1_filler))
                              if s1_filler else float("nan")),
            "n_standing_pref": len(s1_pref),
            "n_filler": len(s1_filler),
            "pass": None,   # no gate attached, per plan/ledger
        },
    }


# ============================================================================
# Artifact schema helpers
# ============================================================================

def make_convo_artifact(conversation_id, probes, rubric="v1"):
    """probes: list of {"probe_id", "candidates": [...]} per the schema
    documented in compute_g1's docstring, PLUS each candidate carries
    "node_class" (fixture label, carried through for G2 pooling).

    v2 additionally permits candidate["s1_mass_per_layer"], a mapping
    {layer_idx_as_string: raw_mass}.  The values are transposed directly
    from ArenaCache.s1_mass(per_layer=True)'s diagnostics dict: unlike the
    registered headline share, each value is raw mass for that candidate in
    that layer.  v1 artifacts have no such key and remain valid inputs.
    """
    return {
        "schema": CONVO_ARTIFACT_SCHEMA,
        "conversation_id": conversation_id,
        "rubric": rubric,
        "probes": probes,
    }


def load_convo_artifacts(artifacts_dir=ARTIFACTS_DIR):
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "convo_*.json")))
    artifacts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema") not in SUPPORTED_CONVO_ARTIFACT_SCHEMAS:
            raise ValueError(
                f"{path}: unexpected schema {data.get('schema')!r}, "
                f"expected one of {sorted(SUPPORTED_CONVO_ARTIFACT_SCHEMAS)!r}")
        artifacts.append(data)
    return artifacts


# ============================================================================
# CPU unit tests (pytest, default collection — no GPU, no model load)
# ============================================================================

# ---- Spearman correctness -------------------------------------------------

def test_spearman_perfect_positive_correlation():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_zero_for_unrelated_hand_case():
    # classic hand-checkable case: ranks [1,2,3] vs [1,3,2] -> rho = 0.5
    a = [1, 2, 3]
    b = [10, 30, 20]
    assert spearman(a, b) == pytest.approx(0.5)


def test_spearman_matches_pearson_on_ranks_hand_computed():
    # a=[3,1,2,4], b=[4,1,3,2] -> ranks(a)=[3,1,2,4], ranks(b)=[4,1,3,2]
    # pearson of those two rank vectors, hand-computed:
    a = [3, 1, 2, 4]
    b = [4, 1, 3, 2]
    ra = np.array([3.0, 1.0, 2.0, 4.0])
    rb = np.array([4.0, 1.0, 3.0, 2.0])
    expected = float(np.corrcoef(ra, rb)[0, 1])
    assert spearman(a, b) == pytest.approx(expected)


def test_spearman_handles_ties_with_average_rank():
    # a has a tie at positions 0,1 (both value 5) -> average rank 1.5 each
    a = [5, 5, 1, 2]
    b = [10, 20, 1, 2]
    # ranks(a) = [1.5, 1.5, 4, 3] wait: sorted a = [1,2,5,5] -> rank(1)=1,
    # rank(2)=2, rank(5)=rank(5)=3.5 (avg of 3,4)
    ra = _rank_average_ties(np.array(a, dtype=np.float64))
    np.testing.assert_allclose(sorted(ra), [1.0, 2.0, 3.5, 3.5])
    # symmetry: spearman(a, b) == spearman(b, a)
    assert spearman(a, b) == pytest.approx(spearman(b, a))


def test_spearman_all_tied_one_side_returns_nan():
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_spearman_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        spearman([1, 2, 3], [1, 2])


def test_spearman_rejects_too_few_points():
    with pytest.raises(ValueError, match="need >= 2"):
        spearman([1], [1])


def test_rank_average_ties_no_ties_is_plain_ranking():
    ranks = _rank_average_ties(np.array([30.0, 10.0, 20.0]))
    np.testing.assert_allclose(ranks, [3.0, 1.0, 2.0])


def test_rank_average_ties_all_equal():
    ranks = _rank_average_ties(np.array([7.0, 7.0, 7.0]))
    np.testing.assert_allclose(ranks, [2.0, 2.0, 2.0])


# ---- top-1 eligibility rule -------------------------------------------------

def test_probe_eligible_when_top_s3_at_or_above_cutoff():
    assert probe_is_eligible({"a": 3.762, "b": 1.0}) is True
    assert probe_is_eligible({"a": 5.0, "b": 1.0}) is True


def test_probe_ineligible_when_top_s3_below_cutoff():
    assert probe_is_eligible({"a": 3.761999, "b": 0.5}) is False
    assert probe_is_eligible({"a": 1.881, "b": 1.0}) is False  # exactly floor


def test_probe_ineligible_when_empty():
    assert probe_is_eligible({}) is False


def test_top1_agreement_hit():
    s3 = {"a": 5.0, "b": 1.0, "c": 0.2}
    sig = {"a": 0.6, "b": 0.3, "c": 0.1}
    assert top1_agreement(sig, s3, ["a", "b", "c"]) == 1


def test_top1_agreement_miss():
    s3 = {"a": 5.0, "b": 1.0, "c": 0.2}
    sig = {"a": 0.1, "b": 0.8, "c": 0.1}
    assert top1_agreement(sig, s3, ["a", "b", "c"]) == 0


def test_top1_agreement_tie_including_correct_answer_counts_as_hit():
    s3 = {"a": 5.0, "b": 1.0}
    sig = {"a": 0.5, "b": 0.5}   # signal ties -> "a" (the s3 winner) is among ties
    assert top1_agreement(sig, s3, ["a", "b"]) == 1


def test_top1_agreement_s3_tie_at_top_counts_signal_matching_either():
    s3 = {"a": 5.0, "b": 5.0, "c": 0.1}
    sig = {"a": 0.1, "b": 0.9, "c": 0.05}
    assert top1_agreement(sig, s3, ["a", "b", "c"]) == 1


# ---- G1 threshold logic (synthetic artifacts) ------------------------------

def _probe(probe_id, rows):
    """rows: list of (candidate_id, ground_truth_grade, s1_mass,
    s2_salience, s3_dep_dlogit, s3_dep_kl, node_class)."""
    return {
        "probe_id": probe_id,
        "candidates": [
            {"candidate_id": cid, "ground_truth_grade": gt, "s1_mass": s1,
             "s2_salience": s2, "s3_dep_dlogit": d, "s3_dep_kl": kl,
             "node_class": nc}
            for cid, gt, s1, s2, d, kl, nc in rows
        ],
    }


def test_g1_pass_case_signal_tracks_s3_perfectly():
    # s1_mass and s2_salience both rank-track s3 and ground truth exactly
    # across 3 probes, all eligible (top s3 >= cutoff) -> should PASS both.
    records = [
        _probe("p1", [
            ("t1", 3, 0.6, 3, 8.0, 1.0, "PROBED_LATER"),
            ("t3", 1, 0.2, 1, 3.0, 0.1, "FILLER"),
            ("t5", 0, 0.05, 0, 0.5, 0.01, "FILLER"),
            ("t7", 2, 0.4, 2, 5.0, 0.5, "FILLER"),
        ]),
        _probe("p2", [
            ("t2", 3, 0.7, 3, 9.0, 1.2, "PROBED_LATER"),
            ("t4", 0, 0.05, 0, 0.3, 0.01, "FILLER"),
            ("t6", 1, 0.2, 1, 2.0, 0.1, "FILLER"),
            ("t8", 2, 0.35, 2, 4.5, 0.4, "FILLER"),
        ]),
        _probe("p3", [
            ("t9", 3, 0.65, 3, 10.0, 1.5, "PROBED_LATER"),
            ("t10", 0, 0.02, 0, 0.2, 0.01, "FILLER"),
            ("t11", 1, 0.15, 1, 2.5, 0.15, "FILLER"),
            ("t12", 2, 0.3, 2, 5.5, 0.6, "FILLER"),
        ]),
    ]
    g1 = compute_g1(records)
    assert g1["s1_mass"]["pass"] is True
    assert g1["s2_salience"]["pass"] is True
    assert g1["s1_mass"]["median_spearman"] == pytest.approx(1.0)
    assert g1["s1_mass"]["top1_agreement"] == pytest.approx(1.0)
    assert g1["s1_mass"]["n_eligible"] == 3


def test_g1_fail_case_signal_uncorrelated_with_s3():
    # s1_mass is inversely related to ground truth / s3 -> should FAIL.
    records = [
        _probe("p1", [
            ("t1", 3, 0.05, 3, 8.0, 1.0, "PROBED_LATER"),
            ("t3", 1, 0.4, 1, 3.0, 0.1, "FILLER"),
            ("t5", 0, 0.6, 0, 0.5, 0.01, "FILLER"),
            ("t7", 2, 0.2, 2, 5.0, 0.5, "FILLER"),
        ]),
        _probe("p2", [
            ("t2", 3, 0.03, 3, 9.0, 1.2, "PROBED_LATER"),
            ("t4", 0, 0.5, 0, 0.3, 0.01, "FILLER"),
            ("t6", 1, 0.35, 1, 2.0, 0.1, "FILLER"),
            ("t8", 2, 0.1, 2, 4.5, 0.4, "FILLER"),
        ]),
    ]
    g1 = compute_g1(records)
    assert g1["s1_mass"]["pass"] is False
    assert g1["s1_mass"]["median_spearman"] < 0


def test_g1_ineligible_probe_excluded_from_top1_but_counted_in_spearman():
    # one probe has top s3 dependence BELOW the eligibility cutoff (no
    # load-bearing winner) -> excluded from top1 agreement denominator,
    # but its Spearman correlation still contributes to the median.
    eligible_probe = _probe("p_hi", [
        ("t1", 3, 0.6, 3, 8.0, 1.0, "PROBED_LATER"),
        ("t2", 0, 0.05, 0, 0.2, 0.01, "FILLER"),
        ("t3", 1, 0.2, 1, 1.0, 0.05, "FILLER"),
        ("t4", 2, 0.3, 2, 2.0, 0.1, "FILLER"),
    ])
    ineligible_probe = _probe("p_lo", [
        # top s3 here is 3.0 < 3.762 cutoff -> ineligible
        ("t5", 3, 0.6, 3, 3.0, 0.2, "PROBED_LATER"),
        ("t6", 0, 0.05, 0, 0.5, 0.05, "FILLER"),
        ("t7", 1, 0.2, 1, 1.0, 0.1, "FILLER"),
        ("t8", 2, 0.3, 2, 2.0, 0.15, "FILLER"),
    ])
    g1 = compute_g1([eligible_probe, ineligible_probe])
    assert g1["s1_mass"]["n_probes_ranked"] == 2      # both ranked
    assert g1["s1_mass"]["n_eligible"] == 1            # only one eligible
    # top1_agreement denominator = 1 (only eligible probe counted)
    assert g1["s1_mass"]["top1_agreement"] == pytest.approx(1.0)


def test_g1_missing_signal_value_excludes_probe_from_ranking():
    # a probe where one candidate's s2_salience is None (QC parse failure)
    # must be excluded entirely from s2_salience's Spearman/top1 for that
    # probe -- never zero-filled or silently dropped-per-candidate.
    rec = _probe("p1", [
        ("t1", 3, 0.6, None, 8.0, 1.0, "PROBED_LATER"),
        ("t2", 0, 0.05, 0, 0.2, 0.01, "FILLER"),
        ("t3", 1, 0.2, 1, 1.0, 0.05, "FILLER"),
        ("t4", 2, 0.3, 2, 2.0, 0.1, "FILLER"),
    ])
    g1 = compute_g1([rec])
    assert g1["s2_salience"]["n_probes_ranked"] == 0
    assert np.isnan(g1["s2_salience"]["median_spearman"])
    assert g1["s1_mass"]["n_probes_ranked"] == 1   # s1_mass unaffected


def test_g1_no_eligible_probes_yields_fail_not_crash():
    ineligible = _probe("p_lo", [
        ("t1", 3, 0.6, 3, 1.0, 0.1, "PROBED_LATER"),
        ("t2", 0, 0.05, 0, 0.5, 0.05, "FILLER"),
        ("t3", 1, 0.2, 1, 0.3, 0.02, "FILLER"),
        ("t4", 2, 0.3, 2, 0.2, 0.01, "FILLER"),
    ])
    g1 = compute_g1([ineligible])
    assert g1["s1_mass"]["n_eligible"] == 0
    assert g1["s1_mass"]["pass"] is False
    assert np.isnan(g1["s1_mass"]["top1_agreement"])


# ---- G2 threshold logic (synthetic pooled candidates) ----------------------

def test_g2_pass_case():
    standing_pref = [{"s2_salience": 3, "s1_mass": 0.01},
                      {"s2_salience": 2, "s1_mass": 0.02},
                      {"s2_salience": 3, "s1_mass": 0.0}]
    filler = [{"s2_salience": 0, "s1_mass": 0.1},
              {"s2_salience": 1, "s1_mass": 0.15},
              {"s2_salience": 0, "s1_mass": 0.12}]
    g2 = compute_g2(standing_pref, filler)
    assert g2["s2_salience"]["pass"] is True
    assert g2["s2_salience"]["median_standing_pref"] == pytest.approx(3.0)
    assert g2["s2_salience"]["median_filler"] == pytest.approx(0.0)
    assert g2["s2_salience"]["delta"] == pytest.approx(3.0)
    # S1 reported, never gated
    assert g2["s1_mass"]["pass"] is None


def test_g2_fail_case_low_median():
    standing_pref = [{"s2_salience": 1, "s1_mass": 0.0},
                      {"s2_salience": 1, "s1_mass": 0.0}]
    filler = [{"s2_salience": 0, "s1_mass": 0.1},
              {"s2_salience": 0, "s1_mass": 0.1}]
    g2 = compute_g2(standing_pref, filler)
    # median standing_pref = 1 < 2 -> fail even though delta (1) meets bar
    assert g2["s2_salience"]["median_standing_pref"] == pytest.approx(1.0)
    assert g2["s2_salience"]["delta"] == pytest.approx(1.0)
    assert g2["s2_salience"]["pass"] is False


def test_g2_fail_case_small_delta():
    standing_pref = [{"s2_salience": 2, "s1_mass": 0.0},
                      {"s2_salience": 2, "s1_mass": 0.0}]
    filler = [{"s2_salience": 2, "s1_mass": 0.1},
              {"s2_salience": 1, "s1_mass": 0.1}]
    g2 = compute_g2(standing_pref, filler)
    # median standing_pref = 2 (meets bar) but delta = 2 - 1.5 = 0.5 < 1
    assert g2["s2_salience"]["median_standing_pref"] == pytest.approx(2.0)
    assert g2["s2_salience"]["delta"] == pytest.approx(0.5)
    assert g2["s2_salience"]["pass"] is False


def test_g2_none_scores_excluded_from_median():
    standing_pref = [{"s2_salience": 3, "s1_mass": 0.0},
                      {"s2_salience": None, "s1_mass": 0.0},   # parse failure
                      {"s2_salience": 3, "s1_mass": 0.0}]
    filler = [{"s2_salience": 0, "s1_mass": 0.1}]
    g2 = compute_g2(standing_pref, filler)
    assert g2["s2_salience"]["n_standing_pref"] == 2   # None excluded
    assert g2["s2_salience"]["median_standing_pref"] == pytest.approx(3.0)


def test_g2_empty_group_yields_fail_not_crash():
    g2 = compute_g2([], [{"s2_salience": 0, "s1_mass": 0.1}])
    assert g2["s2_salience"]["pass"] is False
    assert np.isnan(g2["s2_salience"]["median_standing_pref"])


def test_g2_s1_reported_registered_floor_expectation():
    # registered expectation: S1 on STANDING_PREF sits near the retro
    # floor (near-zero, retrospective signal can't see zero-use nodes)
    # regardless of what S2 does -- this test just proves the reporting
    # path, not any gate (S1 has no gate here).
    standing_pref = [{"s2_salience": 3, "s1_mass": 0.0},
                      {"s2_salience": 3, "s1_mass": 0.001}]
    filler = [{"s2_salience": 0, "s1_mass": 0.3},
              {"s2_salience": 0, "s1_mass": 0.25}]
    g2 = compute_g2(standing_pref, filler)
    assert g2["s1_mass"]["median_standing_pref"] < g2["s1_mass"]["median_filler"]
    assert g2["s1_mass"]["pass"] is None


# ---- artifact schema round-trip --------------------------------------------

def test_make_convo_artifact_round_trips_through_json(tmp_path):
    probes = [_probe("p1", [
        ("t1", 3, 0.6, 3, 8.0, 1.0, "PROBED_LATER"),
        ("t2", 0, 0.05, 0, 0.2, 0.01, "FILLER"),
    ])]
    probes[0]["candidates"][0]["s1_mass_per_layer"] = {
        "0": 0.125,
        "44": 0.375,
    }
    artifact = make_convo_artifact("convo_99_test", probes)
    path = tmp_path / "convo_99_test.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh)
    with open(path, "r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded == artifact
    assert loaded["schema"] == CONVO_ARTIFACT_SCHEMA
    assert loaded["conversation_id"] == "convo_99_test"
    assert loaded["rubric"] == "v1"
    assert loaded["probes"][0]["candidates"][0]["candidate_id"] == "t1"
    assert loaded["probes"][0]["candidates"][0]["s1_mass_per_layer"] == {
        "0": 0.125,
        "44": 0.375,
    }


def test_load_convo_artifacts_rejects_wrong_schema(tmp_path):
    bad = {"schema": "not_the_right_schema_v0", "conversation_id": "x",
           "probes": []}
    path = tmp_path / "convo_bad.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bad, fh)
    with pytest.raises(ValueError, match="unexpected schema"):
        load_convo_artifacts(str(tmp_path))


def test_load_convo_artifacts_reads_multiple_files_sorted(tmp_path):
    for name, cid in (("convo_02_b.json", "b"), ("convo_01_a.json", "a")):
        artifact = make_convo_artifact(cid, [])
        with open(tmp_path / name, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh)
    loaded = load_convo_artifacts(str(tmp_path))
    assert [a["conversation_id"] for a in loaded] == ["a", "b"]


def test_cli_rubric_flag_reaches_mocked_gpu_artifact(monkeypatch):
    """The GPU driver is replaced so this covers CLI plumbing on CPU only."""
    observed = {}

    def fake_run_gpu_driver(convo_n, *, rubric):
        observed["convo_n"] = convo_n
        observed["artifact"] = make_convo_artifact(
            "convo_01_mock", [], rubric=rubric)

    monkeypatch.setattr(sys.modules[__name__], "_run_gpu_driver",
                        fake_run_gpu_driver)

    assert _build_cli_parser().parse_args(
        ["--run-gpu", "--convo", "1"]).rubric == "v1"
    main(["--run-gpu", "--convo", "1", "--rubric", "v2"])

    assert observed["convo_n"] == 1
    assert observed["artifact"]["rubric"] == "v2"


def test_load_convo_artifacts_accepts_mixed_v1_v2_sets(tmp_path):
    """The sealed headline-only v1 receipts and v2 layer receipts coexist.

    This guards the re-run transition: an incomplete GPU batch must remain
    analyzable without inventing per-layer values for its still-v1 legs.
    """
    legacy = {
        "schema": CONVO_ARTIFACT_SCHEMA_V1,
        "conversation_id": "legacy",
        "probes": [_probe("legacy#1", [
            ("1", 3, 0.6, 3, 8.0, 1.0, "PROBED_LATER"),
            ("2", 0, 0.1, 0, 0.2, 0.01, "FILLER"),
        ])],
    }
    modern_probes = [_probe("modern#1", [
        ("1", 3, 0.7, 3, 9.0, 1.2, "PROBED_LATER"),
        ("2", 0, 0.1, 0, 0.3, 0.02, "FILLER"),
    ])]
    for candidate, layer_zero in zip(
            modern_probes[0]["candidates"], (0.8, 0.1)):
        candidate["s1_mass_per_layer"] = {"0": layer_zero}
    modern = make_convo_artifact("modern", modern_probes)

    for name, artifact in (("convo_01_legacy.json", legacy),
                           ("convo_02_modern.json", modern)):
        with open(tmp_path / name, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh)

    loaded = load_convo_artifacts(str(tmp_path))
    assert [artifact["schema"] for artifact in loaded] == [
        CONVO_ARTIFACT_SCHEMA_V1,
        CONVO_ARTIFACT_SCHEMA_V2,
    ]
    assert "s1_mass_per_layer" not in loaded[0]["probes"][0]["candidates"][0]
    assert loaded[1]["probes"][0]["candidates"][0]["s1_mass_per_layer"] == {
        "0": 0.8,
    }


def test_exploratory_diagnostics_do_not_require_v1_layer_data():
    """The new 7a/7b report reads a mixed batch without zero-filling v1."""
    from scripts.grm_importance_diagnostics import build_diagnostics

    legacy = {
        "schema": CONVO_ARTIFACT_SCHEMA_V1,
        "conversation_id": "legacy",
        "probes": [_probe("legacy#1", [
            ("1", 3, 0.6, 3, 8.0, 1.0, "PROBED_LATER"),
            ("2", 0, 0.1, 0, 0.2, 0.01, "FILLER"),
        ])],
    }
    modern_probes = [_probe("modern#1", [
        ("1", 3, 0.7, 3, 9.0, 1.2, "PROBED_LATER"),
        ("2", 0, 0.1, 0, 0.3, 0.02, "FILLER"),
    ])]
    for candidate, layer_zero in zip(
            modern_probes[0]["candidates"], (0.8, 0.1)):
        candidate["s1_mass_per_layer"] = {"0": layer_zero}
    modern = make_convo_artifact("modern", modern_probes)

    diagnostics = build_diagnostics([legacy, modern])
    assert diagnostics["schema"] == "grm_importance_diag_v1"
    assert diagnostics["exploratory"] is True
    assert diagnostics["aggregate"]["s1_mass_vs_s3_dlogit"][
        "n_probes_total"] == 2
    layer_zero = diagnostics["per_layer_s1"]["layers"]["0"]
    assert layer_zero["s1_mass_vs_s3_dlogit"]["n_probes_ranked"] == 1


# ============================================================================
# next_probe_answer_turn — pure fixture-walking, no GPU, no model, no
# engine import. Lead-directed fix: driver crashed on convo_01 turn 20
# ("expected a user-led scripted pair, got role='assistant'") because
# every PROBE turn is followed by a scripted assistant answer turn that
# the original loop tried to consume as the next scripted (user,
# assistant) pair.
# ============================================================================

def _mk_turn(turn_id, role, node_class, text="x"):
    return {"turn_id": turn_id, "role": role, "node_class": node_class,
            "text": text, "node_id": None}


def test_next_probe_answer_turn_consumes_scripted_answer():
    turns = [
        _mk_turn(19, "user", "PROBE", "what's our project codename?"),
        _mk_turn(20, "assistant", "FILLER", "Project HALCYON."),
        _mk_turn(21, "user", "PROBE", "next question"),
    ]
    answer, next_i = next_probe_answer_turn(turns, 0)
    assert answer is not None
    assert answer["turn_id"] == 20
    assert answer["text"] == "Project HALCYON."
    assert next_i == 2   # skip both the probe and its scripted answer


def test_next_probe_answer_turn_probe_at_end_of_file_returns_none():
    turns = [
        _mk_turn(22, "assistant", "FILLER", "prior filler"),
        _mk_turn(23, "user", "PROBE", "last question, no answer follows"),
    ]
    answer, next_i = next_probe_answer_turn(turns, 1)
    assert answer is None
    assert next_i == 2   # past the end of the list


def test_next_probe_answer_turn_back_to_back_probes_returns_none():
    turns = [
        _mk_turn(21, "user", "PROBE", "first probe"),
        _mk_turn(22, "user", "PROBE", "second probe, no answer between"),
    ]
    answer, next_i = next_probe_answer_turn(turns, 0)
    assert answer is None
    assert next_i == 1   # advance past only the first probe


def test_next_probe_answer_turn_raises_on_non_assistant_non_probe_followup():
    # a user-led scripted turn immediately after a probe violates the
    # verified invariant -- must raise, not silently special-case.
    turns = [
        _mk_turn(19, "user", "PROBE", "question"),
        _mk_turn(20, "user", "FILLER", "unexpected user turn"),
    ]
    with pytest.raises(ValueError, match="structural invariant violated"):
        next_probe_answer_turn(turns, 0)


def test_next_probe_answer_turn_raises_if_called_on_non_probe_turn():
    turns = [_mk_turn(1, "user", "FILLER", "not a probe")]
    with pytest.raises(AssertionError, match="called on a non-PROBE turn"):
        next_probe_answer_turn(turns, 0)


def test_next_probe_answer_turn_never_deposited_semantics_via_artifact_field():
    """Driver-level contract check (still pure/no-GPU): the scripted
    answer text lands in the artifact's expected_answer_scripted field,
    never as a candidate or a deposit target. Simulates the driver's
    bookkeeping shape without touching the engine."""
    turns = [
        _mk_turn(19, "user", "PROBE", "q"),
        _mk_turn(20, "assistant", "FILLER", "Project HALCYON."),
    ]
    answer, _ = next_probe_answer_turn(turns, 0)
    probe_record = {
        "probe_id": "convo_x#19",
        "expected_answer_scripted": answer["text"] if answer else None,
        "candidates": [],
    }
    assert probe_record["expected_answer_scripted"] == "Project HALCYON."
    # the scripted answer's turn_id (20) never appears as a candidate_id
    # anywhere -- it was never a graft, never deposited, never graded.
    assert all(c.get("candidate_id") != "20"
              for c in probe_record["candidates"])


def test_fixture_walk_classifies_solo_user_followed_by_probe():
    turns = [
        _mk_turn(1, "user", "FILLER", "solo filler"),
        _mk_turn(2, "user", "PROBE", "probe at EOF"),
    ]
    assert classify_fixture_turns(turns) == [(1, "solo"), (2, "probe")]


def test_fixture_walk_classifies_solo_user_followed_by_user():
    turns = [
        _mk_turn(1, "user", "FILLER", "first user"),
        _mk_turn(2, "user", "FILLER", "second user"),
        _mk_turn(3, "assistant", "FILLER", "paired reply"),
    ]
    assert classify_fixture_turns(turns) == [
        (1, "solo"), (2, "pair"), (3, "pair")]


def test_fixture_walk_classifies_solo_user_at_eof():
    turns = [_mk_turn(1, "user", "FILLER", "solo at EOF")]
    assert classify_fixture_turns(turns) == [(1, "solo")]


def test_scripted_deposit_text_uses_natural_solo_form():
    user = _mk_turn(7, "user", "FILLER", "No reply was scripted.")
    assert scripted_deposit_text(user) == "User: No reply was scripted.\n"


@pytest.mark.parametrize("fixture_path", sorted(
    glob.glob(os.path.join(FIXTURES_DIR, "convo_*.json"))))
def test_real_fixtures_satisfy_probe_answer_invariant(fixture_path):
    """Walk every shipped turn through the driver's structural classifier."""
    with open(fixture_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    turns = data["turns"]
    classified = classify_fixture_turns(turns)  # must not raise
    assert [turn_id for turn_id, _ in classified] == [
        t["turn_id"] for t in turns]
    labels = [label for _, label in classified]
    assert set(labels) <= {"pair", "solo", "probe", "probe-answer"}
    assert labels.count("probe") == len(data["probes"])

    # Lock the complete currently registered shape: convo_02 has the one
    # solo filler (turn 19); all other scripted turns are paired. Every
    # non-EOF probe in all six fixtures has one scripted answer.
    solo_ids = [turn_id for turn_id, label in classified if label == "solo"]
    expected_solos = ([19] if os.path.basename(fixture_path).startswith(
        "convo_02_") else [])
    assert solo_ids == expected_solos
    for idx, t in enumerate(turns):
        if t["node_class"] == "PROBE" and idx + 1 < len(turns):
            assert classified[idx + 1][1] == "probe-answer"


# ============================================================================
# workspace_path / wipe_workspace — pure tempdir, no GPU, no model, no
# GraftRepository import. Lead-directed fix: stale WAL replay across
# repeated --run-gpu invocations against the same workspace path
# reconstructed payload-missing placeholder nodes that crossed the fold
# threshold before the driver fed turn 1 (see wipe_workspace()'s
# docstring for the full named mechanism).
# ============================================================================

def test_workspace_path_is_deterministic_per_convo():
    # same convo -> same path across calls (lead: "prefer the wipe:
    # deterministic paths make artifacts findable" -- no timestamp/pid
    # suffix).
    assert workspace_path(1) == workspace_path(1)
    assert workspace_path(1) != workspace_path(2)


def test_workspace_path_zero_pads_convo_number():
    assert workspace_path(1) == "/tmp/graftrepo_g1g2_convo_01"
    assert workspace_path(6) == "/tmp/graftrepo_g1g2_convo_06"
    assert workspace_path(12) == "/tmp/graftrepo_g1g2_convo_12"


def test_wipe_workspace_removes_existing_directory_and_contents(tmp_path):
    ws = tmp_path / "graftrepo_g1g2_convo_01"
    ws.mkdir()
    (ws / "manifest.json").write_text("{}")
    wal_dir = ws / "wal"
    wal_dir.mkdir()
    (wal_dir / "0001.wal").write_text('{"type": "NODE_META"}\n')
    nodes_dir = ws / "nodes"
    nodes_dir.mkdir()
    (nodes_dir / "0000.npz").write_bytes(b"stale payload bytes")

    assert ws.exists()
    wipe_workspace(str(ws))
    assert not ws.exists()


def test_wipe_workspace_is_a_noop_on_nonexistent_path(tmp_path):
    ws = tmp_path / "never_existed"
    assert not ws.exists()
    wipe_workspace(str(ws))          # must not raise
    assert not ws.exists()


def test_wipe_workspace_leaves_sibling_paths_untouched(tmp_path):
    ws = tmp_path / "graftrepo_g1g2_convo_01"
    ws.mkdir()
    (ws / "manifest.json").write_text("{}")
    sibling = tmp_path / "graftrepo_g1g2_convo_02"
    sibling.mkdir()
    (sibling / "manifest.json").write_text('{"marker": "sibling"}')

    wipe_workspace(str(ws))

    assert not ws.exists()
    assert sibling.exists()
    assert (sibling / "manifest.json").read_text() == '{"marker": "sibling"}'


def test_wipe_then_recreate_produces_a_fresh_empty_directory(tmp_path):
    """The driver's actual sequence: wipe, THEN construct a fresh
    GraftRepository at the same path (which recreates the directory tree
    itself, per _ensure_repo_dirs -- this test only proves the wipe half:
    after wiping, the path is gone, so anything reading it next sees no
    stale manifest/wal, not a mix of old and new state)."""
    ws = tmp_path / "graftrepo_g1g2_convo_01"
    ws.mkdir()
    wal_dir = ws / "wal"
    wal_dir.mkdir()
    (wal_dir / "0001.wal").write_text(
        '{"type": "NODE_META", "node_id": 0}\n')

    wipe_workspace(str(ws))
    assert not ws.exists()

    # simulate the repo re-creating its own directory structure fresh
    ws.mkdir()
    (ws / "wal").mkdir()
    assert list((ws / "wal").iterdir()) == []   # no leftover WAL records


# ============================================================================
# --analyze : CPU-only, reads real artifacts written by the GPU driver leg.
# ============================================================================

def _pool_node_class(artifacts, node_class):
    """Flatten every candidate of the given node_class across every probe
    of every convo artifact into one list of candidate dicts (for G2's
    STANDING_PREF-vs-FILLER pooling)."""
    out = []
    for art in artifacts:
        for probe in art["probes"]:
            for cand in probe["candidates"]:
                if cand.get("node_class") == node_class:
                    out.append(cand)
    return out


def _all_probe_records(artifacts):
    out = []
    for art in artifacts:
        for probe in art["probes"]:
            out.append(probe)
    return out


def run_analysis(artifacts_dir=ARTIFACTS_DIR):
    artifacts = load_convo_artifacts(artifacts_dir)
    if not artifacts:
        raise SystemExit(
            f"no convo_*.json artifacts found under {artifacts_dir} — run "
            f"the GPU driver leg first (--run-gpu --convo N for N in 1..6)")
    records = _all_probe_records(artifacts)
    g1 = compute_g1(records)
    standing_pref = _pool_node_class(artifacts, "STANDING_PREF")
    filler = _pool_node_class(artifacts, "FILLER")
    g2 = compute_g2(standing_pref, filler)

    print(f"Loaded {len(artifacts)} convo artifacts, {len(records)} probes "
          f"total.", flush=True)
    print(f"Eligibility cutoff (2x G0b noise floor): "
          f"{ELIGIBILITY_CUTOFF:.3f}", flush=True)
    print(flush=True)
    print("=== G1 (signal agreement) ===", flush=True)
    for signal in ("s1_mass", "s2_salience"):
        r = g1[signal]
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {signal:14s} median_spearman={r['median_spearman']:.4f} "
              f"(>= {G1_MIN_MEDIAN_SPEARMAN})  "
              f"top1_agreement={r['top1_agreement']:.4f} "
              f"(>= {G1_MIN_TOP1_AGREEMENT}, n_eligible={r['n_eligible']}"
              f"/{r['n_probes_ranked']} ranked)  -> {status}", flush=True)
    print(flush=True)
    print("=== G2 (prospective discriminator) ===", flush=True)
    s2r = g2["s2_salience"]
    status = "PASS" if s2r["pass"] else "FAIL"
    print(f"  s2_salience  median(STANDING_PREF)={s2r['median_standing_pref']:.4f} "
          f"(>= {G2_MIN_MEDIAN_STANDING_PREF})  "
          f"median(FILLER)={s2r['median_filler']:.4f}  "
          f"delta={s2r['delta']:.4f} (>= {G2_MIN_STANDING_PREF_MINUS_FILLER}) "
          f"n={s2r['n_standing_pref']}/{s2r['n_filler']}  -> {status}",
          flush=True)
    s1r = g2["s1_mass"]
    print(f"  s1_mass      median(STANDING_PREF)={s1r['median_standing_pref']:.4f}  "
          f"median(FILLER)={s1r['median_filler']:.4f}  "
          f"n={s1r['n_standing_pref']}/{s1r['n_filler']}  "
          f"(registered expectation of failure; no gate)", flush=True)

    verdict = {
        "schema": VERDICT_SCHEMA,
        "n_convos": len(artifacts),
        "n_probes": len(records),
        "eligibility_cutoff": ELIGIBILITY_CUTOFF,
        "g1": g1,
        "g2": g2,
    }
    print(flush=True)
    print(json.dumps(verdict), flush=True)
    return verdict


# ============================================================================
# --run-gpu --convo N : GPU driver leg. NOT executed by pytest collection
# (guarded by __main__ + explicit flags) and NOT executed by this authoring
# agent per work order. The lead runs, once per conversation (1..6):
#
#   python3 tests/test_grm_importance_g1g2.py --run-gpu --convo 1
#   ... --convo 2
#   ... --convo 6
#
# then, CPU-only, once all six artifacts exist:
#
#   python3 tests/test_grm_importance_g1g2.py --analyze
# ============================================================================

def next_probe_answer_turn(turns, probe_index):
    """Pure fixture-walking helper (no GPU, no engine): given the full
    `turns` list and the 0-based index of a PROBE-class turn within it,
    return (scripted_answer_turn_or_None, next_index).

    Lead-verified fixture structure (all 6 convos, 2026-07-16): every
    PROBE turn is immediately followed by exactly ONE scripted assistant
    turn carrying the fixture's reference answer (e.g. probe "what's our
    project codename?" -> scripted "Project HALCYON."). That pair is
    NEVER deposited: probe Q&A is exactly the retrieval-only wake-turn
    class the deposit-hygiene law (core/graft_arena.py _attempt's
    "retrieval hygiene" comment, kind="recall") forbids depositing, and no
    fixture relevance map ever grades a probe-turn or probe-answer
    turn_id (WO-4 validator: relevance keys must precede the probe turn).
    The ONE exception is a probe that is the LAST turn in the file (no
    turn follows it at all — true end of conversation, not a violation of
    the one-scripted-assistant-follows rule); that case returns
    (None, probe_index + 1).

    Any other shape (next turn absent AND not EOF, next turn not an
    assistant turn, next turn itself a second PROBE) is a structural
    violation of the verified invariant and raises loudly rather than
    guessing — per instruction, special-casing an unverified fixture
    shape is out of scope; this must be reported, not silently handled.
    """
    t = turns[probe_index]
    assert t["node_class"] == "PROBE", (
        f"next_probe_answer_turn called on a non-PROBE turn "
        f"{t['turn_id']} (class={t['node_class']!r})")
    if probe_index + 1 >= len(turns):
        return None, probe_index + 1   # probe is the last turn in the file
    nxt = turns[probe_index + 1]
    if nxt["node_class"] == "PROBE":
        # back-to-back probes: no scripted answer sits between them. Not
        # observed in any of the 6 shipped fixtures (verified by lead +
        # re-checked here), but the work order named it explicitly as a
        # shape to handle rather than assume away.
        return None, probe_index + 1
    if nxt["role"] != "assistant":
        raise ValueError(
            f"probe turn {t['turn_id']}: expected the following turn to "
            f"be a scripted assistant answer or another PROBE, got "
            f"turn {nxt['turn_id']} role={nxt['role']!r} "
            f"class={nxt['node_class']!r} — fixture structural invariant "
            f"violated, not special-casing this")
    return nxt, probe_index + 2


def next_scripted_deposit_turn(turns, user_index):
    """Return (assistant_half_or_None, next_index) for a non-PROBE user.

    A following assistant forms the repository's usual paired exchange.
    A following user (including a PROBE) or EOF makes this a solo user
    deposit; the following user remains unconsumed for the next walk step.
    """
    t = turns[user_index]
    assert t["node_class"] != "PROBE", (
        f"next_scripted_deposit_turn called on PROBE turn {t['turn_id']}")
    assert t["role"] == "user", (
        f"expected a scripted user turn at {t['turn_id']}, "
        f"got role={t['role']!r}")
    if user_index + 1 >= len(turns):
        return None, user_index + 1
    nxt = turns[user_index + 1]
    if nxt["role"] == "assistant":
        return nxt, user_index + 2
    if nxt["role"] == "user":
        return None, user_index + 1
    raise ValueError(
        f"turn {nxt['turn_id']} has unsupported role={nxt['role']!r}")


def classify_fixture_turns(turns):
    """Classify every turn using the exact structural walk of the driver."""
    classified = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t["node_class"] == "PROBE":
            assert t["role"] == "user", (
                f"PROBE turn {t['turn_id']} must have role='user', "
                f"got {t['role']!r}")
            classified.append((t["turn_id"], "probe"))
            answer, i = next_probe_answer_turn(turns, i)
            if answer is not None:
                classified.append((answer["turn_id"], "probe-answer"))
            continue
        assistant, next_i = next_scripted_deposit_turn(turns, i)
        label = "pair" if assistant is not None else "solo"
        classified.append((t["turn_id"], label))
        if assistant is not None:
            classified.append((assistant["turn_id"], label))
        i = next_i

    walked_ids = [turn_id for turn_id, _ in classified]
    fixture_ids = [t["turn_id"] for t in turns]
    assert walked_ids == fixture_ids, (
        f"fixture walk did not classify every turn exactly once: "
        f"walked={walked_ids}, fixture={fixture_ids}")
    return classified


def scripted_deposit_text(user_turn, assistant_turn=None):
    """Format a paired or solo scripted deposit exactly as ArenaCache does."""
    assert user_turn["role"] == "user"
    text = f"User: {user_turn['text']}\n"
    if assistant_turn is not None:
        assert assistant_turn["role"] == "assistant"
        text += f"Assistant: {assistant_turn['text']}\n"
    return text


def _load_fixture(convo_n):
    paths = sorted(glob.glob(os.path.join(FIXTURES_DIR, "convo_*.json")))
    matches = [p for p in paths
               if os.path.basename(p).startswith(f"convo_{convo_n:02d}_")]
    if not matches:
        raise SystemExit(f"no fixture found for convo {convo_n} under "
                          f"{FIXTURES_DIR}")
    with open(matches[0], "r", encoding="utf-8") as fh:
        return json.load(fh), matches[0]


def workspace_path(convo_n):
    """The GraftRepository workspace dir for one convo's GPU driver leg.
    Deterministic (not timestamped) on purpose — lead's instruction:
    "prefer the wipe: deterministic paths make artifacts findable."""
    return f"/tmp/graftrepo_g1g2_convo_{convo_n:02d}"


def wipe_workspace(path):
    """Delete a GraftRepository workspace directory if it exists.

    NAMED FAILURE MODE (lead, 2026-07-16, empirically confirmed via
    monkeypatch instrumentation on leg 1): repeated --run-gpu invocations
    against the SAME workspace path accumulate a wal/ directory across
    attempts (autosave=False does NOT stop WAL writes — WAL append is a
    separate durability mechanism from autosave/flush). A crashed prior
    run's WAL, replayed on the NEXT invocation's GraftRepository.__init__
    (core/graft_repository.py: no manifest.json -> _read_wal() ->
    _rehydrate_from_wal()), reconstructs WAL-recovered PLACEHOLDER nodes:
    kind="turn", ntok=0, retired=False, h=None, payload_pending=True,
    host_payload=None, no .npz on disk. These placeholders are real turn-
    kind, non-retired, non-live grafts to _active()/_foldable(), so they
    silently count toward TURNS_HIGH. Leg 1's crash: the repo BOOTED with
    11 such ghosts already past threshold, before this driver fed turn 1
    -- idle() folded them on the very first call, consolidate() called
    _ensure_h() on payload-missing placeholders (no RAM copy, no disk
    file to load), which can only return None, and the unguarded
    `self.grafts[i]["h"][li]` indexing crashed. This is DRIVER-SIDE
    workspace hygiene, not a mount/sweep bug in this file's own logic
    (repo-side seam: fold selection not excluding payload-missing
    placeholders -- routed elsewhere, not owned here).

    Each --run-gpu invocation must therefore start from a clean workspace
    with ZERO grafts. The path is a /tmp workspace fully owned by this
    harness (never shared with another program), so deleting it outright
    is safe -- no confirmation, no data of value outside a rerunnable GPU
    leg."""
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _run_gpu_driver(convo_n, rubric="v1"):
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tensor_cuda as tc
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from core.graft_repository import GraftRepository
    from tokenizers import Tokenizer as HFTok

    fixture, fixture_path = _load_fixture(convo_n)
    print(f"driving {fixture_path} ({fixture['conversation_id']})",
          flush=True)

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)

    # WARM-UP (ledger, "G0a first run RED"): the first forward pass of a
    # process differs from all subsequent warm passes at bf16-noise scale.
    # A throwaway forward before anything measured, matching the pattern
    # every other GPU gate in this program uses.
    with tc.no_grad():
        m(np.array([[1, 2, 3]], dtype=np.int64), kv_caches=None,
          position_offset=0, last_token_only=True)

    # WORKSPACE HYGIENE (lead, 2026-07-16 — see wipe_workspace()'s
    # docstring for the full named mechanism: stale WAL replay across
    # repeated invocations against the same path reconstructs payload-
    # missing placeholder nodes that silently cross the fold threshold
    # before this driver ever feeds turn 1). Wipe unconditionally before
    # construction — deterministic path, always clean start.
    ws_path = workspace_path(convo_n)
    wipe_workspace(ws_path)

    repo = GraftRepository(
        m, lambda t: tok.encode(t).ids, lambda ids: tok.decode(ids),
        ws_path,
        autosave=False, librarian_mode="deferred",
        sink_text="<conversation>\n", arena_width=512, route_layer=44,
        topk=3, live_turns=2, ephemeral=False, recency_mounts=2,
        s2_salience_enabled=True, rubric=rubric)
    n_boot_nodes = len(repo.arena.grafts)
    print(f"post-construction node count: {n_boot_nodes}", flush=True)
    assert n_boot_nodes == 0, (
        f"workspace hygiene regression: {ws_path} booted with "
        f"{n_boot_nodes} node(s) instead of 0 -- the wipe above did not "
        f"produce a clean workspace (see wipe_workspace()'s docstring for "
        f"the stale-WAL-replay failure mode this guards against). Loud "
        f"setup failure instead of a deep librarian crash, per lead "
        f"instruction.")
    arena = repo.arena
    for L in m.layers:
        L.self_attn.live_shift = arena.live_shift
        # S1 tap only fires inside the ABSORBED DECODE branch
        # (core/minicpm3_tc.py:226-257, `if absorbed_decode and L==1 and
        # B==1 and kv_cache is not None...`); the telemetry.numpy()/
        # _accumulate_telemetry() call sits INSIDE that branch (line
        # 243-252) and is never reached on the generic
        # scaled_dot_product_attention path. Every other GPU gate in this
        # program sets this (test_graft_e4_arena.py:31,
        # test_grm_importance_telemetry.py:292) before measuring.
        L.self_attn.absorbed_decode = True

    turns = fixture["turns"]
    probes_by_turn = {p["probe_turn_id"]: p for p in fixture["probes"]}
    turn_by_id = {t["turn_id"]: t for t in turns}

    # turn_id -> deposited graft index. Scripted (user, assistant) pairs are
    # one node (add_turn -> arena.feed("User: ...\nAssistant: ...\n")); a
    # scripted user without an assistant half is one natural solo node
    # (arena.feed("User: ...\n")). Map every deposited fixture turn_id so
    # either shape can be a graded probe candidate.
    turn_id_to_graft = {}

    def deposit_scripted(user_turn, asst_turn=None):
        before_len = len(arena.grafts)
        if asst_turn is None:
            before = repo._snapshot_state()
            # add_turn(user, assistant) always emits the Assistant scaffold;
            # feed() is the lowest public deposit path that can represent a
            # genuinely absent assistant half. Complete the same repository
            # bookkeeping sequence as GRMRuntime.add_turn() around that feed.
            arena.feed(scripted_deposit_text(user_turn))
            repo._set_new_node_provenance(before, "exchange_span")
            extracted = repo._extract_from_new_turns(
                before, context={"event": "add_turn",
                                 "user_text": user_turn["text"],
                                 "assistant_text": None})
            result = repo.runtime._finish_turn_event(
                "add_turn", before, extraction=extracted, autosave=False)
            repo._queue_s2_pending()
            new_nodes = result.new_nodes
        else:
            repo.add_turn(user_turn["text"], asst_turn["text"])
            new_nodes = repo.runtime.last_result.new_nodes
        assert len(new_nodes) == 1 and new_nodes[0] == before_len, (
            f"expected exactly one new node at {before_len}, got "
            f"{new_nodes}")
        gidx = new_nodes[0]
        turn_id_to_graft[user_turn["turn_id"]] = gidx
        if asst_turn is not None:
            turn_id_to_graft[asst_turn["turn_id"]] = gidx
        repo.idle(max_jobs=1)   # fires S2 immediately at deposit time
        return gidx

    all_probe_records = []
    i = 0
    n = len(turns)
    while i < n:
        t = turns[i]
        if t["node_class"] == "PROBE":
            probe = probes_by_turn[t["turn_id"]]
            candidate_turn_ids = sorted(int(k) for k in probe["relevance"])
            picks = sorted({turn_id_to_graft[tid]
                            for tid in candidate_turn_ids})
            print(f"  PROBE turn={t['turn_id']} candidates(turn_ids)="
                  f"{candidate_turn_ids} -> picks(graft_idx)={picks}",
                  flush=True)

            arena.set_telemetry(True)
            reply, attempt_info = arena._attempt(
                probe["question"], picks, ngen=48, deposit=False,
                stops=arena.stop_sequences)
            # The headline S1 share remains the registered all-layer mean.
            # v2 also records the per-layer raw diagnostics for successor
            # 7a only; no layer selection happens inside this driver.
            s1, s1_per_layer = arena.s1_mass(per_layer=True)
            print(f"    reply={reply[:80]!r} mounts={attempt_info['mounts']} "
                  f"s1_mass={s1}", flush=True)
            arena.set_telemetry(False)

            # Snapshot the live conversation state exactly as it stands
            # after the probe's own (un-deposited) turn landed in the live
            # cache — the S3 counterfactual sweep below runs on an
            # INDEPENDENT fresh mini-cache (mirrors
            # test_grm_importance_counterfactual.py's teacher_forced_logits
            # closure) and must not perturb this state, so the
            # conversation can resume at the next scripted turn.
            resume_snap = (arena.caches, arena.pos, list(arena.live_segs),
                           arena.cur_mounts, arena.cur_mount_n,
                           list(arena.grafts))

            prompt_ids = arena.encode(arena._format_step_prompt(
                probe["question"]))
            reply_ids = arena.encode(reply)

            def teacher_forced_logits(sub_picks):
                arena.caches, arena.pos, arena.live_segs = None, 0, []
                arena.cur_mounts, arena.cur_mount_n = [], 0
                arena._ensure_h(sub_picks)
                from core import kv_graft
                mounts = ([{"h": arena.sink_h}]
                         + [arena.grafts[i] for i in sub_picks])
                _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
                inj = []
                for li in range(len(arena.m.layers)):
                    inj.append({key: np.concatenate(
                        [_np(g["h"][li][key]) for g in mounts], axis=dim)
                        for key, dim in arena.PAYLOAD})
                arena._set_injection_host(inj)
                arena.cur_mounts = sub_picks
                arena.cur_mount_n = sum(arena.grafts[i]["ntok"]
                                        for i in sub_picks)
                arena._commit_native_mount(sub_picks, arena.cur_mount_n)
                for L in arena.m.layers:
                    L.self_attn.live_shift = arena.live_shift
                full_ids = prompt_ids + reply_ids
                with tc.no_grad():
                    lg, _ = arena.m(np.array([full_ids], dtype=np.int64),
                                    kv_caches=None,
                                    position_offset=arena.live_shift,
                                    last_token_only=False)
                kv_graft.clear_injection(arena.m)
                logits = lg.numpy()[0].astype(np.float32)
                return logits[len(prompt_ids) - 1:
                             len(prompt_ids) - 1 + len(reply_ids)]

            # warm-up throwaway forward at the SAME shape class before
            # capturing the full-set reference (G0a first-run law).
            _ = teacher_forced_logits(picks)
            full_logits = teacher_forced_logits(picks)

            candidates = []
            for tid in candidate_turn_ids:
                cidx = turn_id_to_graft[tid]
                minus_picks = minus_node_picks(picks, cidx)
                minus_logits = teacher_forced_logits(minus_picks)
                dep = dependence(full_logits, minus_logits)
                s2_val = (arena.grafts[cidx].get("metadata", {})
                         .get("importance", {}).get("s2_salience"))
                s1_val = s1.get(cidx)
                # ArenaCache returns {layer_idx: {graft_idx: raw_mass}}.
                # Store its candidate-oriented transpose without inventing
                # a zero for a layer that did not expose telemetry.
                s1_per_layer_val = {
                    str(layer_idx): raw_by_graft[cidx]
                    for layer_idx, raw_by_graft in s1_per_layer.items()
                    if cidx in raw_by_graft
                }
                node_class = turn_by_id[tid]["node_class"]
                candidates.append({
                    "turn_id": tid,
                    "candidate_id": str(tid),
                    "node_class": node_class,
                    "ground_truth_grade": probe["relevance"][str(tid)],
                    "s1_mass": s1_val,
                    "s1_mass_per_layer": s1_per_layer_val,
                    "s2_salience": s2_val,
                    "s3_dep_dlogit": dep["mean_abs_dlogit"],
                    "s3_dep_kl": dep["mean_kl"],
                })
                print(f"    candidate turn={tid} node={cidx} "
                      f"class={node_class} gt={probe['relevance'][str(tid)]} "
                      f"s1={s1_val} s2={s2_val} "
                      f"s3_dlogit={dep['mean_abs_dlogit']:.4f} "
                      f"s3_kl={dep['mean_kl']:.4f}", flush=True)

            # Consume the fixture's scripted reference-answer turn (if any)
            # that immediately follows this probe. Recorded in the
            # artifact for visibility/debugging but NEVER deposited —
            # probe Q&A is the retrieval-only wake-turn class the deposit-
            # hygiene law forbids depositing (core/graft_arena.py
            # _attempt's "retrieval hygiene" comment), and no fixture
            # relevance map grades a probe-pair turn_id (WO-4 validator).
            answer_turn, next_i = next_probe_answer_turn(turns, i)
            expected_answer_scripted = (
                answer_turn["text"] if answer_turn is not None else None)

            all_probe_records.append({
                "probe_id": f"{fixture['conversation_id']}#{t['turn_id']}",
                "expected_answer_scripted": expected_answer_scripted,
                "candidates": candidates,
            })

            # restore the pre-counterfactual live state so the scripted
            # conversation can resume. The un-deposited probe turn (and
            # the un-deposited, never-fed scripted answer) leave no trace
            # here — resume_snap was taken right after the probe's own
            # generation, before the S3 sweep, and the scripted answer
            # turn was never fed into the live cache at all.
            (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
             arena.cur_mount_n) = (resume_snap[0], resume_snap[1],
                                   list(resume_snap[2]), resume_snap[3],
                                   resume_snap[4])
            arena.grafts[:] = resume_snap[5]
            i = next_i
            continue

        # Scripted pair or solo user -> one deposit; S2 fires now.
        asst, next_i = next_scripted_deposit_turn(turns, i)
        deposit_scripted(t, asst)
        i = next_i

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    artifact = make_convo_artifact(fixture["conversation_id"],
                                   all_probe_records, rubric=rubric)
    out_path = os.path.join(
        ARTIFACTS_DIR, f"{fixture['conversation_id']}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"wrote {out_path}", flush=True)
    print(json.dumps({"schema": CONVO_ARTIFACT_SCHEMA,
                      "conversation_id": fixture["conversation_id"],
                      "n_probes": len(all_probe_records)}), flush=True)
    print("DONE", flush=True)
    return artifact


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="GRM-IMPORTANCE G1/G2 gate driver + analysis")
    parser.add_argument("--run-gpu", action="store_true",
                        help="run the GPU driver leg for one conversation "
                             "(loads MiniCPM3, no co-resident model loads "
                             "— lead-only, per work order GPU discipline)")
    parser.add_argument("--convo", type=int, default=None,
                        help="conversation number 1-6 (required with "
                             "--run-gpu)")
    parser.add_argument("--analyze", action="store_true",
                        help="run the CPU-only G1/G2 analysis over "
                             "artifacts already written by --run-gpu legs")
    parser.add_argument("--rubric", choices=("v1", "v2"), default="v1",
                        help="S2 salience rubric for --run-gpu (default: v1)")
    return parser


def main(argv=None):
    args = _build_cli_parser().parse_args(argv)
    if args.run_gpu:
        if args.convo is None:
            print("--run-gpu requires --convo N (1-6)", flush=True)
            sys.exit(1)
        _run_gpu_driver(args.convo, rubric=args.rubric)
    elif args.analyze:
        verdict = run_analysis()
        all_pass = (all(v["pass"] for v in verdict["g1"].values())
                   if verdict["g1"] else False)
        g2_pass = verdict["g2"]["s2_salience"]["pass"]
        sys.exit(0 if (all_pass or g2_pass) else 1)
    else:
        print("No action flag given. CPU unit tests run under pytest "
              "(`python3 -m pytest tests/test_grm_importance_g1g2.py`). "
              "GPU driver: python3 tests/test_grm_importance_g1g2.py "
              "--run-gpu --convo N (N=1..6). Analysis (CPU-only, after all "
              "six artifacts exist): python3 tests/test_grm_importance_"
              "g1g2.py --analyze", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
