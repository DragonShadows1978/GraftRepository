"""Lead amendment 2: CPU diagnostic doubles, never a replacement reader.

Prior art: local GRM LSR-P2B E2E fixtures and runtime lifecycle FakeSliceArena
(GRM contributors, 2026). Borrowed: replace numerical model/payload boundaries
while executing repository, admission and serving code. Ours: observe C7's
exact input strings, real _attempt and consolidate calls. No prior art known
to me for this exact diagnostic composition; no new memory algorithm.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re

import numpy as np

from core.graft_arena import ArenaCache
from core.graft_repository import GraftRepository
from scripts import grm_e2e_session as e2e

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_c7/diagnosis_lead_2'


def record(name, value):
    directory = Path(os.environ.get('GRM_C7_DIAG_OUT', OUT))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (name + '.json')).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


class Codec:
    """Reversible CPU tokenizer; token lengths are NOT GPT-OSS geometry."""
    def __init__(self):
        self.words = ['']
        self.ids = {'': 0}

    def encode(self, text):
        # Prior art: Python re token splitting (Python contributors, 2026).
        # Preserve whitespace exactly; only a numerical boundary test double.
        result = []
        for word in re.findall(r'\S+|\s+', text):
            if word not in self.ids:
                self.ids[word] = len(self.words)
                self.words.append(word)
            result.append(self.ids[word])
        return result

    def decode(self, ids):
        return ''.join(self.words[int(i)] for i in ids)


class Model:
    """Deterministic model stub: copy an explicitly visible value, else UNKNOWN.

    Prior art: local GRM scripted reader doubles (2026); no prior art known
    to me for this exact regex. This is test stimulus, not an inference fix.
    Never reads probe.expected. It records the input before choosing output.
    """
    config = SimpleNamespace(num_layers=1, hidden_dim=512, kv_lora_rank=512,
                             qk_rope_head_dim=0)

    def __init__(self, codec):
        self.codec = codec
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))]
        self.calls = []
        self.injected = ''
        self.remaining = []
        self.fold_output = None

    def extend_rope(self, length):
        pass

    def __call__(self, ids, kv_caches=None, position_offset=0, **kwargs):
        text = self.codec.decode(ids[0])
        if kv_caches is None:
            if self.fold_output is not None:
                answer = self.fold_output
            else:
                # The source must be visible to this fake reader; the fixture's
                # expected answer is neither passed in nor inspected here.
                visible = self.injected + '\n' + text
                matches = re.findall(r'current [\w-]+(?: [\w-]+)? value is ([\w-]+)', visible)
                answer = (matches[-1] if matches else 'UNKNOWN') + '<|end|>'
            self.remaining = self.codec.encode(answer)
        token = self.remaining.pop(0) if self.remaining else self.codec.encode('…')[0]
        self.calls.append({'input': text, 'position_offset': position_offset,
                           'live_shift': self.layers[0].self_attn.live_shift,
                           'initial': kv_caches is None, 'output_token': token,
                           'injected': self.injected})
        logits = np.zeros((1, 1, len(self.codec.words)), dtype=np.float32)
        logits[0, 0, token] = 1
        return SimpleNamespace(numpy=lambda: logits), {'tokens': position_offset + len(ids[0])}


class CPUArena(ArenaCache):
    """Numerical/payload seams only; routing, admission and attempts inherited."""
    PAYLOAD = (('tok', 0),)
    VALS_PER_TOK_LAYER = 1

    def _harvest(self, ids, **kwargs):
        return [{'tok': np.asarray(ids, dtype=np.int64)}]

    def _node_key(self, text, h_host=None):
        return np.ones(512, dtype=np.float32) / np.sqrt(512)

    def _probe_key(self, text):
        return self._node_key(text)

    def deposit(self, text, capture_pin=None):
        ids = self.encode(text)
        self.grafts.append({'h': self._harvest(ids), 'cent': self._node_key(text),
                            'ntok': len(ids), 'text': text})
        return len(self.grafts) - 1

    def _set_injection_host(self, inj):
        self.m.injected = self.decode(inj[0]['tok'])

    def _set_inject(self, att, blk):
        self.m.injected = self.decode(blk['tok'])

    def _clear_transients(self):
        pass

    def _cache_len(self):
        return self.pos + self.n_sink + self.cur_mount_n

    def evict(self):
        # No finite-cache simulator: all diagnostic prompts fit the test's
        # declared unbounded CPU cache. No residency/position quality claim.
        return 0

    def pack_node(self, h):
        return {'tok': h[0]['tok']}

    def unpack_node(self, z):
        return [{'tok': z['tok']}]


def repository(path, monkeypatch):
    import core.graft_arena as ga
    # Prior art: unittest/pytest in-process dependency replacement; local
    # lifecycle fixtures (GRM, 2026). Patch only allocation/no-grad plumbing;
    # no CUDA tensor, checkpoint, tokenizer, backend or model is loaded.
    monkeypatch.setattr(ga.tc, 'no_grad', nullcontext)
    monkeypatch.setattr(ga.tc, 'cat', lambda values, dim: np.concatenate(values, axis=dim))
    monkeypatch.setenv('GRM_DEMAND_NGH', '0')
    monkeypatch.setenv('GRM_RT1_RULE', '1')
    monkeypatch.setenv('GRM_LSR_FIXES', '1')
    monkeypatch.setenv('GRM_GQA_CUDA_ROUTE', '0')
    codec = Codec()
    model = Model(codec)
    repo = GraftRepository(model, codec.encode, codec.decode, str(path),
        autosave=False, native_auto=False, arena_cls=CPUArena,
        arena_width=96, route_layer=0, ephemeral=True, recency_mounts=0,
        sink_text=e2e.HARMONY_SINK, prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS, revision_resolution=True,
        decisive_admission=True, route_backend='python')
    repo.arena._rs3_seat_explicit = False
    return repo
