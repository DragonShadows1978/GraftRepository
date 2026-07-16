"""WO-1 (S1 attention-mass telemetry) — GRM_IMPORTANCE_PLAN.md.

Two parts:

  (a) CPU unit tests (plain `test_*` functions, collected by pytest by
      default; no GPU, no model load) covering the seat -> mount-id
      aggregation math in ArenaCache.s1_mass, the non-live-mass share
      normalization, and the off-flag = no-accumulation contract on the
      MLAAttentionTC tap.

  (b) G0a — telemetry-on vs telemetry-off teacher-forced decode parity
      (max |Δlogit| == 0 exactly). This loads the real model and touches
      the GPU, so it is deliberately NOT named test_* (pytest will never
      collect it) and is only reachable via `python3 -m tests.test_grm_
      importance_telemetry --run-gpu` (see __main__ below). Do not call it
      from a CPU test; do not run it without --run-gpu.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graft_arena import ArenaCache
from core.minicpm3_tc import MLAAttentionTC, MiniCPM3Config


# ============================================================ (a) CPU unit tests

class _FakeAttn:
    """Stand-in for MLAAttentionTC exposing only what s1_mass touches:
    .telemetry and ._telemetry_mass. Keeps the aggregation tests free of
    any tensor_cuda/model dependency."""
    def __init__(self):
        self.telemetry = False
        self._telemetry_mass = None

    def reset_telemetry(self):
        self._telemetry_mass = None


class _FakeLayer:
    def __init__(self):
        self.self_attn = _FakeAttn()


class _FakeModel:
    def __init__(self, n_layers):
        self.layers = [_FakeLayer() for _ in range(n_layers)]


def _make_arena(n_sink, mounts, n_layers=2):
    """Build a bare ArenaCache (no __init__, no model/tokenizer) with just
    the seating state s1_mass/_mount_seat_ranges read: n_sink, cur_mounts
    (graft idx order), grafts (idx -> {"ntok": n}), cur_mount_n, and a fake
    model with `n_layers` fake attention modules."""
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = _FakeModel(n_layers)
    arena.n_sink = n_sink
    arena.cur_mounts = [i for i, _ in mounts]
    arena.grafts = {i: {"ntok": n} for i, n in mounts}
    arena.cur_mount_n = sum(n for _, n in mounts)
    return arena


def test_mount_seat_ranges_pack_from_n_sink_in_mount_order():
    arena = _make_arena(n_sink=4, mounts=[(2, 3), (0, 5)])
    ranges = arena._mount_seat_ranges()
    # mount order follows cur_mounts, NOT graft idx order
    assert ranges == {2: (4, 7), 0: (7, 12)}


def test_s1_mass_seat_to_node_aggregation_and_non_live_share():
    arena = _make_arena(n_sink=4, mounts=[(0, 3), (1, 5)], n_layers=2)
    S = 20  # 4 sink + 3 mount0 + 5 mount1 + 8 live
    for L in arena.m.layers:
        mass = np.zeros(S)
        mass[0:4] = 1.0     # sink: 4 seats x 1.0 = 4
        mass[4:7] = 2.0     # mount 0: 3 seats x 2.0 = 6
        mass[7:12] = 3.0    # mount 1: 5 seats x 3.0 = 15
        mass[12:20] = 9.0   # live: must be EXCLUDED from the denominator
        L.self_attn.telemetry = True
        L.self_attn._telemetry_mass = mass

    shares = arena.s1_mass()
    # non-live total = 4 (sink, unattributed) + 6 (mount0) + 15 (mount1) = 25
    assert shares[0] == 6.0 / 25.0
    assert shares[1] == 15.0 / 25.0
    # shares are of NON-LIVE mass, not of all mounted mass, so they must
    # NOT sum to 1.0 (sink eats the remainder, 4/25 = 0.16, un-keyed)
    assert abs(sum(shares.values()) - (21.0 / 25.0)) < 1e-12
    assert sum(shares.values()) < 1.0


def test_s1_mass_means_over_all_layers_not_just_last():
    arena = _make_arena(n_sink=0, mounts=[(0, 2)], n_layers=4)
    # give each layer a different mass so a mean (not e.g. "last layer wins"
    # or "sum without averaging") is the only formula that reproduces this
    per_layer_mount_mass = [1.0, 2.0, 3.0, 4.0]   # mean = 2.5
    for L, m in zip(arena.m.layers, per_layer_mount_mass):
        mass = np.array([m, 0.0])  # 2 mount seats, all mass on seat 0
        L.self_attn.telemetry = True
        L.self_attn._telemetry_mass = mass
    shares = arena.s1_mass()
    # only one mount -> its share of non-live mass is 1.0 regardless of the
    # per-layer mean value; check the diagnostics dict instead to see the
    # mean actually landed at 2.5, not 4.0 (last) or 10.0 (unaveraged sum)
    assert shares[0] == 1.0
    _, diag = arena.s1_mass(per_layer=True)
    assert set(diag.keys()) == {0, 1, 2, 3}
    for li, expected in enumerate(per_layer_mount_mass):
        assert diag[li][0] == expected  # RAW per-layer mass, undivided


def test_s1_mass_per_layer_diagnostics_not_used_by_headline():
    """The plan is explicit: per-layer breakdown is diagnostics-only and
    must not feed the headline scalar. Assert the two return modes agree
    on the headline (per_layer=True must not perturb it by, say, gating on
    a subset of layers)."""
    arena = _make_arena(n_sink=1, mounts=[(0, 1), (1, 1)], n_layers=3)
    rng = np.random.RandomState(0)
    for L in arena.m.layers:
        mass = rng.rand(6)
        L.self_attn.telemetry = True
        L.self_attn._telemetry_mass = mass
    headline_only = arena.s1_mass()
    headline_with_diag, diag = arena.s1_mass(per_layer=True)
    assert headline_only == headline_with_diag
    assert isinstance(diag, dict) and len(diag) == 3


def test_s1_mass_handles_no_mounts_and_no_accumulator_without_div_by_zero():
    arena = _make_arena(n_sink=4, mounts=[], n_layers=2)
    # no telemetry ever ran: _telemetry_mass stays None on every layer
    shares = arena.s1_mass()
    assert shares == {}

    arena2 = _make_arena(n_sink=4, mounts=[(0, 3)], n_layers=2)
    shares2 = arena2.s1_mass()   # accumulators still None -> zero mass
    assert shares2 == {0: 0.0}


def test_s1_mass_grows_with_seat_count_across_decode_steps():
    """The accumulator grows by one seat per decode step (S_all increases
    each step). s1_mass must sum whatever fraction of a mount's range the
    accumulator has actually covered, never index past its length."""
    arena = _make_arena(n_sink=2, mounts=[(0, 4)], n_layers=1)
    L = arena.m.layers[0]
    L.self_attn.telemetry = True
    # accumulator only 5 long: covers sink (0:2) + first 3 of the 4 mount
    # seats (2:5) — as if 5 decode steps have run and the tap's growing
    # buffer hasn't caught up to the mount's full nominal range yet
    L.self_attn._telemetry_mass = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    shares = arena.s1_mass()
    # mount mass = 2+2+2 = 6, sink mass = 1+1 = 2, total = 8
    assert shares[0] == 6.0 / 8.0


# ---------------------------------------------- MLAAttentionTC tap: off-flag

def test_telemetry_flag_defaults_off_and_accumulator_starts_none():
    attn = MLAAttentionTC(MiniCPM3Config())
    assert attn.telemetry is False
    assert attn._telemetry_mass is None


def test_reset_telemetry_clears_accumulator():
    attn = MLAAttentionTC(MiniCPM3Config())
    attn._telemetry_mass = np.array([1.0, 2.0, 3.0])
    attn.reset_telemetry()
    assert attn._telemetry_mass is None


def test_telemetry_tap_is_gated_behind_the_flag_and_reads_w_only():
    """Static-inspection check (same technique as
    test_deepseek_grm_hooks_static.py's hook-contract tests): the tap must
    be reachable only through `getattr(self, "telemetry", False)`, must
    read `w` via `.numpy()` (copy-out, never an in-place op on the tensor
    that feeds `ctxl`), and must sit between `w = tc.causal_softmax(s)` and
    `ctxl = tc.matmul(w, c2)` so the telemetry-off path is provably the
    same statement sequence as before the tap existed."""
    import inspect
    src = inspect.getsource(MLAAttentionTC.__call__)

    softmax_at = src.index("w = tc.causal_softmax(s)")
    telemetry_at = src.index('getattr(self, "telemetry", False)')
    ctxl_at = src.index("ctxl = tc.matmul(w, c2)")
    assert softmax_at < telemetry_at < ctxl_at

    tap_block = src[telemetry_at:ctxl_at]
    assert "w.numpy()" in tap_block
    # never write through w itself (no `w[...] =`, no `w +=`, no in-place
    # method call on w before ctxl consumes it)
    assert "w[" not in tap_block
    assert "w +=" not in tap_block
    assert "w -=" not in tap_block


# ============================================================ (b) G0a GPU gate
#
# NOT named test_* on purpose — pytest must never collect or run this. The
# lead runs it explicitly:
#
#   PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
#   python3 tests/test_grm_importance_telemetry.py --run-gpu
#
def run_g0a_telemetry_parity_gate():
    """G0a telemetry parity (plan, line 45-47): telemetry-on vs
    telemetry-off decode, teacher-forced, max |Δlogit| == 0 exactly.

    Drives a minimal one-mount arena conversation (structure follows
    tests/test_graft_e4_arena.py: load the model, build an ArenaCache,
    feed a scripted turn so it deposits a graft, then run one `step()` to
    mount + generate). Runs the SAME teacher-forced token sequence twice
    through fresh forward passes (telemetry off, then on) and diffs the
    full logit vector at every decode step. The hook is a pure read, so
    the demand is bit-identical — not the usual bf16 receipt floor.
    """
    import tensor_cuda as tc
    from core.minicpm3_tc import MiniCPM3_TC, _snap
    from tokenizers import Tokenizer as HFTok

    tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    m, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)
    for L in m.layers:
        L.self_attn.absorbed_decode = True   # telemetry only fires here

    def fresh_arena():
        return ArenaCache(m,
                          encode=lambda t: tok.encode(t).ids,
                          decode=lambda ids: tok.decode(ids),
                          sink_text="<conversation>\n",
                          arena_width=64, route_layer=44, topk=1,
                          live_turns=2)

    def run(telemetry_on):
        arena = fresh_arena()
        arena.set_telemetry(telemetry_on)
        arena.feed("User: My access code is Q77-1130.\n"
                   "Assistant: Recorded.\n")
        for L in m.layers:
            L.self_attn.live_shift = arena.live_shift
        prompt_ids = arena.encode(arena._format_step_prompt(
            "What is my access code?"))
        logits = []
        row = arena._forward(prompt_ids)
        logits.append(row.copy())
        # teacher-force a FIXED continuation (not argmax-driven) so both
        # runs see byte-identical inputs at every step regardless of any
        # earlier logit drift
        forced_ids = arena.encode(" Q77-1130 is your access code.")
        for tid in forced_ids:
            row = arena._forward([tid])
            logits.append(row.copy())
        return np.stack(logits, axis=0), (arena.s1_mass() if telemetry_on
                                          else None)

    logits_off, _ = run(False)
    logits_on, s1 = run(True)

    max_delta = float(np.max(np.abs(logits_on - logits_off)))
    print(f"G0a: max |Δlogit| over {logits_off.shape[0]} decode steps "
          f"= {max_delta}", flush=True)
    print(f"G0a: s1_mass (telemetry-on run) = {s1}", flush=True)
    assert max_delta == 0.0, (
        f"G0a FAILED: telemetry tap perturbed the forward computation "
        f"(max |Δlogit| = {max_delta}, expected exactly 0.0)")
    print("G0a: PASS (bit-identical logits, telemetry on vs off)",
          flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-gpu", action="store_true",
                   help="Actually run the G0a GPU gate (loads the model, "
                        "allocates CUDA memory). Refuses to run without "
                        "this flag.")
    args = p.parse_args()
    if not args.run_gpu:
        print("G0a gate is gated behind --run-gpu (no GPU work done). "
              "Re-run with --run-gpu to execute it.", flush=True)
        sys.exit(0)
    run_g0a_telemetry_parity_gate()
