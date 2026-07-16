"""GRM-IMPORTANCE S3 (docs/GRM_IMPORTANCE_PLAN.md, WO-2) — counterfactual-
unmount ground truth. S3 dependence(node) = teacher-forced replay of a
reply with the mounted set MINUS that node, vs the full mounted set;
metric = mean |Δlogit| over reply tokens (also report KL). House law
(measured repo law, not a guess): any cache-equivalence comparison MUST be
teacher-forced — generation-based A/B is garbage past the first greedy
divergence, so this harness never samples/generates during measurement,
only replays fixed token ids through the model.

Two layers:
  1. CPU unit tests (pytest, run by default, no GPU, no model load): pure
     numpy math for the Δlogit/KL metrics, and the mounted-set-minus-node
     construction logic that drives ArenaCache.swap() picks.
  2. G0b GATE (script-style, gated behind `--run-gpu`; the lead runs this,
     never the authoring agent): a scripted single-turn conversation on
     MiniCPM3 MLA with TWO mounted grafts — one load-bearing (the reply's
     answer depends on it) and one decoy (irrelevant filler). Asserts
     dependence(load-bearing) >> dependence(decoy) and prints the decoy's
     value (the measured S3 noise floor) plus the load-bearing/decoy ratio
     (the dynamic range) as a machine-readable JSON line. No numeric pass
     threshold is hardcoded beyond that ordering — these are the numbers
     G1/G2 register their thresholds from (SCRIBE floor pattern).

Reuses: core/kv_graft.py harvest/inject primitives and core/graft_arena.py
ArenaCache.deposit()/swap()/_forward() cache-surgery mounting, the same
teacher-forced all_logits() pattern as tests/test_graft_mla_gate.py G1/G2
and the SCRIPTED/PROBES arena-driving pattern of tests/test_graft_e4_arena.py.
"""
import argparse
import json
import os
import sys

import numpy as np
import pytest


# --------------------------------------------------------------------------
# Metrics — pure numpy, no engine/model dependency. These are the exact
# reductions the G0b gate applies to real per-token logit arrays, unit
# tested here against synthetic arrays so the math is verified on CPU
# before any GPU time is spent.
# --------------------------------------------------------------------------

def mean_abs_delta_logit(logits_a, logits_b):
    """mean over reply tokens of max|Δlogit| per token (S3 registered
    metric: "mean |Δlogit| over reply tokens"). logits_a/b: (T, V) arrays,
    one row per reply token, full vocab. Per-token reduction is max over
    vocab (the single most-moved logit that token), then mean over T —
    matches the max|logit diff| convention used throughout the repo's
    teacher-forced gates (test_graft_mla_gate.py, test_scribe_g0.py),
    aggregated across the reply span instead of reported per-token.
    """
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected (T, V), got {a.shape}")
    per_token_max = np.max(np.abs(a - b), axis=-1)
    return float(per_token_max.mean())


def _log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    s = x - m
    lse = np.log(np.exp(s).sum(axis=-1, keepdims=True))
    return s - lse


def mean_kl(logits_a, logits_b):
    """mean over reply tokens of KL(P_a || P_b), P = softmax(logits).
    "Full-set" distribution is the reference (a); reported alongside
    mean_abs_delta_logit per the plan's "also report KL"."""
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected (T, V), got {a.shape}")
    log_pa = _log_softmax(a)
    log_pb = _log_softmax(b)
    pa = np.exp(log_pa)
    kl_per_token = np.sum(pa * (log_pa - log_pb), axis=-1)
    return float(kl_per_token.mean())


def dependence(full_logits, minus_logits):
    """S3 dependence(node) bundle: (mean |Δlogit|, mean KL) between the
    full-mounted-set replay and the mounted-set-minus-node replay, both
    teacher-forced over the same reply token ids."""
    return {
        "mean_abs_dlogit": mean_abs_delta_logit(full_logits, minus_logits),
        "mean_kl": mean_kl(full_logits, minus_logits),
    }


# --------------------------------------------------------------------------
# Mounted-set-minus-node construction — the picks-list logic that drives
# ArenaCache.swap(). Pure list/int logic, independent of the engine: given
# the full set of mounted graft indices and a target node, returns the
# ordered pick list with that node removed. ArenaCache.swap() takes an
# explicit ordered list of graft indices (see core/graft_arena.py `swap`),
# so this is the exact argument the G0b gate passes per counterfactual.
# --------------------------------------------------------------------------

def minus_node_picks(mounted, node):
    """mounted: ordered list of graft indices currently seated (as tracked
    by ArenaCache.cur_mounts). node: the graft index to counterfactually
    unmount. Returns a NEW sorted list with `node` removed — swap() sorts
    its own cache-surgery internally but callers (E4-arena, this harness)
    consistently pass sorted picks, so we match that convention.

    Raises if node is not in mounted — a counterfactual on an absent node
    is a caller bug, not a silent no-op (would score dependence() on two
    identical replays and read the noise floor as "not dependent" for
    free, corrupting the floor)."""
    if node not in mounted:
        raise ValueError(f"node {node} not in mounted set {mounted}")
    return sorted(i for i in mounted if i != node)


def all_minus_one_picks(mounted):
    """The full counterfactual sweep: {node: minus_node_picks(mounted, node)
    for node in mounted}. What the G0b gate (and later G1) iterates."""
    mounted = list(mounted)
    return {node: minus_node_picks(mounted, node) for node in mounted}


# ============================================================================
# CPU unit tests (pytest, default collection — no GPU, no model load)
# ============================================================================

def test_mean_abs_delta_logit_zero_for_identical_arrays():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 128)).astype(np.float32)
    assert mean_abs_delta_logit(a, a.copy()) == 0.0


def test_mean_abs_delta_logit_matches_hand_computed_value():
    a = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 2.5], [3.0, 0.5, 0.0]], dtype=np.float32)
    # row0: |diffs| = [0, 0, 0.5] -> max 0.5 ; row1: |diffs| = [0, 0.5, 0] -> max 0.5
    assert mean_abs_delta_logit(a, b) == pytest.approx(0.5)


def test_mean_abs_delta_logit_rejects_shape_mismatch():
    a = np.zeros((4, 10), dtype=np.float32)
    b = np.zeros((4, 11), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        mean_abs_delta_logit(a, b)


def test_mean_abs_delta_logit_rejects_wrong_rank():
    a = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(T, V\)"):
        mean_abs_delta_logit(a, a)


def test_mean_kl_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 64)).astype(np.float32)
    assert mean_kl(a, a.copy()) == pytest.approx(0.0, abs=1e-9)


def test_mean_kl_matches_scalar_reference_two_class():
    # two-class case: KL(P||Q) has a closed form we can hand-check.
    a = np.array([[0.0, 0.0]], dtype=np.float64)          # P = [0.5, 0.5]
    b = np.array([[0.0, np.log(3.0)]], dtype=np.float64)  # Q = [0.25, 0.75]
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    expected = float(np.sum(p * np.log(p / q)))
    assert mean_kl(a, b) == pytest.approx(expected, rel=1e-9)


def test_mean_kl_is_nonnegative_and_asymmetric_in_general():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(8, 32)).astype(np.float32)
    b = rng.normal(size=(8, 32)).astype(np.float32)
    kl_ab = mean_kl(a, b)
    kl_ba = mean_kl(b, a)
    assert kl_ab >= -1e-9
    assert kl_ba >= -1e-9
    assert kl_ab != pytest.approx(kl_ba)


def test_dependence_bundle_reports_both_metrics():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 16)).astype(np.float32)
    b = a + rng.normal(scale=0.1, size=(4, 16)).astype(np.float32)
    d = dependence(a, b)
    assert set(d.keys()) == {"mean_abs_dlogit", "mean_kl"}
    assert d["mean_abs_dlogit"] > 0.0
    assert d["mean_kl"] > 0.0


def test_dependence_bundle_zero_when_replays_identical():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(3, 20)).astype(np.float32)
    d = dependence(a, a.copy())
    assert d["mean_abs_dlogit"] == 0.0
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)


def test_minus_node_picks_removes_only_target():
    mounted = [2, 5, 7]
    assert minus_node_picks(mounted, 5) == [2, 7]
    assert minus_node_picks(mounted, 2) == [5, 7]
    assert minus_node_picks(mounted, 7) == [2, 5]


def test_minus_node_picks_does_not_mutate_input():
    mounted = [4, 1, 9]
    _ = minus_node_picks(mounted, 1)
    assert mounted == [4, 1, 9]


def test_minus_node_picks_returns_sorted_output():
    mounted = [9, 1, 4]
    assert minus_node_picks(mounted, 9) == [1, 4]


def test_minus_node_picks_raises_on_absent_node():
    with pytest.raises(ValueError, match="not in mounted set"):
        minus_node_picks([1, 2, 3], 99)


def test_all_minus_one_picks_covers_every_mounted_node():
    mounted = [3, 8, 12]
    sweep = all_minus_one_picks(mounted)
    assert set(sweep.keys()) == {3, 8, 12}
    assert sweep[3] == [8, 12]
    assert sweep[8] == [3, 12]
    assert sweep[12] == [3, 8]


def test_all_minus_one_picks_handles_single_mount():
    sweep = all_minus_one_picks([42])
    assert sweep == {42: []}


# ============================================================================
# G0b GATE — script-style, GPU, gated behind --run-gpu. NOT executed by
# pytest collection (guarded by __main__ + explicit flag) and NOT executed
# by this authoring agent per work order. The lead runs:
#
#   python3 tests/test_grm_importance_counterfactual.py --run-gpu
#
# Loads MiniCPM3 via core.minicpm3_tc.MiniCPM3_TC, drives an ArenaCache
# exactly as tests/test_graft_e4_arena.py does, deposits one load-bearing
# graft + one decoy graft, mounts both, teacher-forces the scripted reply
# with the full set and with each set-minus-one, and reports S3
# dependence() for both nodes plus the derived floor/ratio.
# ============================================================================

def _run_g0b_gate():
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tensor_cuda as tc
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from core.graft_arena import ArenaCache
    from tokenizers import Tokenizer as HFTok

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)

    arena = ArenaCache(m,
                       encode=lambda t: tok.encode(t).ids,
                       decode=lambda ids: tok.decode(ids),
                       sink_text="<conversation>\n",
                       arena_width=256, route_layer=44, topk=3, live_turns=2)
    print(f"arena: sink={arena.n_sink} seats, width={arena.width}, "
          f"live_shift={arena.live_shift}", flush=True)

    # Load-bearing graft: the reply's fact comes from here.
    LOAD_BEARING = ("MAINTENANCE BULLETIN. The coolant manifold on line 4 "
                     "was replaced at shift change; torque spec 48 Nm; "
                     "part lot VX-2291. Sign-off: foreman Ilsa Brandt.")
    # Decoy: irrelevant content, same rough length/register, unrelated to
    # the probe below.
    DECOY = ("FIELD NOTE. The tagged osprey (band K-557) fished the north "
              "weir at dawn, three successful strikes in nine minutes. "
              "Wind calm. Next census on the 14th.")

    lb_idx = arena.deposit(LOAD_BEARING)
    decoy_idx = arena.deposit(DECOY)
    mounted = sorted([lb_idx, decoy_idx])
    print(f"deposited: load-bearing={lb_idx} decoy={decoy_idx}", flush=True)

    # Scripted single turn: user asks the load-bearing question, reply is
    # FIXED (teacher-forced) — never generated. This is the reply whose
    # per-token logits get compared full-set vs minus-node.
    USER = "What is the part lot of the replaced coolant manifold?"
    REPLY = " The part lot is VX-2291."
    prompt_ids = arena.encode(arena._format_step_prompt(USER))
    reply_ids = arena.encode(REPLY)

    def teacher_forced_logits(picks):
        """Mount `picks` on a FRESH cache (bootstrap path), then
        teacher-force [prompt + reply] in one forward, last_token_only=False,
        and return the reply span's per-token logits (T, V). Fresh cache
        per call keeps counterfactuals independent (no cross-call cache
        leakage) — matches test_graft_mla_gate.py's all_logits() re-prefill
        pattern, adapted to route through ArenaCache's mount plumbing."""
        arena.caches, arena.pos, arena.live_segs = None, 0, []
        arena.cur_mounts, arena.cur_mount_n = [], 0
        arena._ensure_h(picks)
        from core import kv_graft
        mounts = [{"h": arena.sink_h}] + [arena.grafts[i] for i in picks]
        _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
        inj = []
        for li in range(len(arena.m.layers)):
            inj.append({key: np.concatenate([_np(g["h"][li][key])
                                             for g in mounts], axis=dim)
                        for key, dim in arena.PAYLOAD})
        arena._set_injection_host(inj)
        arena.cur_mounts = picks
        arena.cur_mount_n = sum(arena.grafts[i]["ntok"] for i in picks)
        arena._commit_native_mount(picks, arena.cur_mount_n)
        for L in arena.m.layers:
            L.self_attn.live_shift = arena.live_shift
        full_ids = prompt_ids + reply_ids
        with tc.no_grad():
            lg, _ = arena.m(np.array([full_ids], dtype=np.int64),
                            kv_caches=None, position_offset=arena.live_shift,
                            last_token_only=False)
        kv_graft.clear_injection(arena.m)
        logits = lg.numpy()[0].astype(np.float32)
        # reply-token logits are the PREDICTIONS at each position preceding
        # a reply token, i.e. positions [len(prompt_ids)-1 : -1]
        return logits[len(prompt_ids) - 1: len(prompt_ids) - 1 + len(reply_ids)]

    # Warm-up throwaway forward, SAME shape class as the measured passes
    # (full mounted set, same prompt+reply length) before capturing side A.
    # Ledger 2026-07-16 "G0a first run RED": the first forward of a process
    # differs by <=0.5 logit (bf16-noise scale) from all subsequent warm
    # runs, which are bit-identical to each other — a same-process A/B must
    # warm up before capturing side A or the reference carries cold-run
    # pollution at exactly the decoy noise-floor scale being measured.
    _ = teacher_forced_logits(mounted)

    full_logits = teacher_forced_logits(mounted)

    results = {}
    for label, node in (("load_bearing", lb_idx), ("decoy", decoy_idx)):
        minus_picks = minus_node_picks(mounted, node)
        minus_logits = teacher_forced_logits(minus_picks)
        d = dependence(full_logits, minus_logits)
        results[label] = d
        print(f"S3 dependence[{label}] (node={node}, minus={minus_picks}): "
              f"mean|Δlogit|={d['mean_abs_dlogit']:.6f}  "
              f"mean_KL={d['mean_kl']:.6f}", flush=True)

    floor = results["decoy"]["mean_abs_dlogit"]
    signal = results["load_bearing"]["mean_abs_dlogit"]
    ratio = signal / floor if floor > 0 else float("inf")

    ordering_ok = signal > floor
    print(f"\nG0b: load-bearing {'>>' if ordering_ok else 'NOT>>'} decoy "
          f"(ratio={ratio:.2f}x)", flush=True)

    report = {
        "schema": "grm_importance_s3_g0b_v1",
        "load_bearing": results["load_bearing"],
        "decoy": results["decoy"],
        "noise_floor_mean_abs_dlogit": floor,
        "noise_floor_mean_kl": results["decoy"]["mean_kl"],
        "dynamic_range_ratio": ratio,
        "ordering_assertion_pass": ordering_ok,
    }
    print(json.dumps(report), flush=True)

    assert ordering_ok, (
        f"S3 dynamic range collapsed: load-bearing dependence "
        f"({signal:.6f}) did not exceed decoy noise floor ({floor:.6f})")

    print("DONE", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRM-IMPORTANCE S3 counterfactual-unmount harness")
    parser.add_argument("--run-gpu", action="store_true",
                        help="run the G0b GPU gate (loads MiniCPM3, no "
                             "co-resident model loads — lead-only, per "
                             "work order GPU discipline)")
    args = parser.parse_args()
    if not args.run_gpu:
        print("No --run-gpu flag: nothing to do here. CPU unit tests run "
              "under pytest (`python3 -m pytest "
              "tests/test_grm_importance_counterfactual.py`). To run the "
              "G0b GPU gate: python3 tests/test_grm_importance_counterfactual.py "
              "--run-gpu", flush=True)
        sys.exit(0)
    _run_g0b_gate()
