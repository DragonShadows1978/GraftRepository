"""SCOUT-FIX-9 host-only reproduction of the P1 GPU smoke r2 admission crash.

Loads the SAVED r2 session repository (artifacts/grm_p1/gpu_session/repository)
with a host-only GQA arena — real repository, routing, native store and
admission code; stubbed harvest/payload/probe-key numerics — and re-runs
`decisive_admission_profile` under GRM_ADMISSION_RULE=margin_first.

Prior art: the C7 CPU numerical doubles (scripts/grm_c7_diagnose.py) and the
LSR-P2B E2E fixtures (GRM contributors, 2026) — reused wholesale for the
"replace numerics, execute real control flow" pattern. Ours is only loading a
LIVED GPU session's repository into that pattern so a crash recorded on the
GPU can be reproduced and fixed without one. No prior art known to me for this
exact composition; unverified against the wider literature (no network in this
sandbox) — lead to check.

NOT a model-quality or recall measurement: the probe key is a deterministic
RNG draw, not a GPT-OSS harvest, so the ABSOLUTE ranking differs from the GPU
log. What it reproduces is the DIVERGENCE between the native router and the
Python A-DEC reconstruction, which is a property of the stored route keys.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_scout_fix9'
SESSION = ROOT / 'artifacts/grm_p1/gpu_session/repository'
NATIVE_LIB = ROOT / 'artifacts/grm_wc1_astra/native_build/libgrm_runtime.so'
QUESTION = 'recap the five biggest decisions we made'

ENV = {
    'GRM_PERSISTENT_BOAT': '0', 'GRM_CAPTURE_PIN': 'live',
    'GRM_SEAT_NEAR_LIVE': '1', 'GRM_LSR_FIXES': '1', 'GRM_RT1_RULE': '1',
    'GRM_DEMAND_NGH': '0', 'GRM_GQA_CUDA_ROUTE': '0',
    'GRM_GRAFT_STORAGE_BITS': '8', 'GRM_ROUTE_QUERY_LEX': '1',
    'GRM_PROBE_LADDER': '1', 'GRM_SUP_RESOLVE': '1', 'GRM_ADM_DECISIVE': '1',
    'GRM_ADMISSION_RULE': 'margin_first',
}


def _seeded(text, shape):
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def build():
    os.environ.setdefault('GRM_RUNTIME_LIB', str(NATIVE_LIB))
    os.environ.update(ENV)
    import core.graft_arena as ga
    from core.graft_arena import GQAArenaCache
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c7_diagnose import Codec, Model
    from core.gpt_oss20b_tc import GptOss20BConfig, GptOssAttentionTC

    ga.tc.no_grad = nullcontext
    ga.tc.cat = lambda values, dim: np.concatenate(values, axis=dim)

    class ReplayArena(GQAArenaCache):
        """Host-only GQA arena: real routing/admission, stubbed numerics."""
        PAYLOAD = (('k', 2), ('v', 2))
        POSITION_LAW = 'rope_full_yarn'

        def _harvest(self, ids, layer_filter=None, max_layers=None):
            n = len(ids[0]) if np.ndim(ids) > 1 else len(ids)
            return [{'k': np.zeros((1, 8, n, 64), np.float32),
                     'v': np.zeros((1, 8, n, 64), np.float32)}]

        def _probe_key(self, text):
            return _seeded(text, (64, 4, 64))

        def _node_key(self, text, h_host=None):
            return _seeded(text, (8, 4, 64))

        def _clear_transients(self):
            pass

    # The dialect wall compares the MODEL CLASS NAME; this double stands in
    # for GPT-OSS-20B's metadata only (no weights, no GPU, no forward).
    class GptOss20B_TC(Model):
        pass

    work = Path(tempfile.mkdtemp(prefix='fix9-repro-')) / 'repository'
    shutil.copytree(SESSION, work)
    codec = Codec()
    model = GptOss20B_TC(codec)
    cfg = GptOss20BConfig(layer_types=('full_attention',) * 24)
    model.config = cfg
    model.layers = [SimpleNamespace(self_attn=GptOssAttentionTC(cfg, i))
                    for i in range(24)]
    repo = GraftRepository(
        model, codec.encode, codec.decode, str(work), autosave=False,
        arena_cls=ReplayArena, arena_width=96, route_layer=1, ephemeral=True,
        recency_mounts=0, sink_text=e2e.HARMONY_SINK,
        prompt_template=e2e.harmony_turn, stop_sequences=e2e.HARMONY_STOPS,
        revision_resolution=True, decisive_admission=True,
        route_backend='auto', native_enabled=True, native_auto=True)
    return repo


def observe(repo):
    from core.grm_admission import (AdmissionPolicyError,
                                    decisive_admission_profile)
    a = repo.arena
    pk = a._probe_key(QUESTION)
    eligible = [int(i) for i in a._route_cand_base()]
    native_raw = [int(v) for v in a.native_store.route_gqa(
        np.asarray(pk, dtype=np.float32), [], topk=len(a.grafts))]
    python_raw = sorted(
        eligible, key=lambda i: (-a._cent_score(pk, a.grafts[i]), i))
    row = {
        'question': QUESTION,
        'eligible': eligible,
        'native_raw_order': native_raw,
        'python_raw_order': python_raw,
        'orders_agree': native_raw == python_raw,
        'cent_scores': {str(i): float(a._cent_score(pk, a.grafts[i]))
                        for i in eligible},
        'descent_key_counts': {
            str(i): len(a.grafts[i].get('child_cents') or ())
            for i in eligible},
        'keyless_nodes': [int(i) for i, g in enumerate(a.grafts)
                          if g.get('cent') is None],
    }
    try:
        profile = decisive_admission_profile(
            a, QUESTION, exclude=(), route_limit=6)
    except AdmissionPolicyError as exc:
        row['raised'] = 'AdmissionPolicyError'
        row['message'] = str(exc)
        row['ranking'] = None
        row['rank_plan'] = None
    else:
        row['raised'] = None
        row['message'] = None
        row['ranking'] = [int(v) for v in profile['ranking']]
        row['rank_plan'] = [int(v) for v in profile['rank_plan']]
        row['policy_branch'] = str(profile['policy_branch'])
        row['route_backend'] = str(profile['route_backend'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['red', 'green'])
    args = ap.parse_args()
    repo = build()
    try:
        row = observe(repo)
    finally:
        repo.close()
    row['phase'] = args.phase
    row['gpu_executed'] = False
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.phase + '.json')
    path.write_text(json.dumps(row, indent=2, sort_keys=True,
                               ensure_ascii=False) + '\n')
    print(json.dumps({k: row[k] for k in (
        'phase', 'raised', 'message', 'orders_agree', 'ranking', 'rank_plan')},
        ensure_ascii=False))
    # RED is EXPECTED to raise; GREEN is expected not to.
    if args.phase == 'red':
        return 0 if row['raised'] else 1
    return 0 if not row['raised'] and row['orders_agree'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
