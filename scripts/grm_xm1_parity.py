#!/usr/bin/env python3
"""XM1 one-cell runner; dry-run imports no model or GPU library.

Prior art: GRM contributors (2026), RS3 capture-pin/constant-delta seating,
RS4 live_band_bounds/split_full_row/mean_over_positions (imported verbatim),
LT1 loader seam and create-only resumable receipts. This is instrumentation,
not a new attention algorithm. RoPE: Su et al. 2021; MLA: DeepSeek-AI 2024,
unverified — lead to check: RoFormer 2104.09864, DeepSeek-V2 2405.04434.
Ours: model dispatch, source/cell bindings and hard cross-model barrier.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.grm_xm1_registration import OUT, MODELS, ARMS, read, sha, create
from scripts.grm_rs4_row_split import (live_band_bounds, split_full_row,
    mean_over_positions, partition_sums_to_one, subbands_sum_to_live)

REG = OUT/'registration.json'
IMPL = OUT/'implementation_pins.json'
AMENDMENT = OUT/'registration_amendment_1.json'
XM2_REG = ROOT/'artifacts/grm_xm2/registration.json'
XM2_AMENDMENT = ROOT/'artifacts/grm_xm2/registration_amendment_1.json'

class XM1Error(RuntimeError):
    pass


def xm2_registration():
    # Prior art: GRM XM1/LT1 immutable amendment overlays (2026), reused.
    # The original XM2 registration remains immutable, including failed caps.
    base = read(XM2_REG)
    amendment = read(XM2_AMENDMENT)
    if amendment['registration_sha256'] != sha(XM2_REG):
        raise XM1Error('XM2 amendment has wrong registration binding')
    effective = {**base, **amendment['overrides']}
    effective['pins'] = {**base['pins'], **amendment['pins']}
    return effective


def validate_registration():
    reg = read(REG)
    for rel, digest in reg['pins'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(ROOT/rel) != digest:
            raise XM1Error(f'registration source drift: {rel}')
    impl = read(IMPL)
    if impl['registration_sha256'] != sha(REG):
        raise XM1Error('implementation amendment has wrong registration binding')
    # Prior art: LT1 immutable receipt/source bindings (GRM, 2026).
    # Amendment 1 overlays only explicitly rebound files; originals stay intact.
    amendment = read(AMENDMENT)
    if (amendment['registration_sha256'] != sha(REG)
            or amendment['previous_implementation_sha256'] != sha(IMPL)):
        raise XM1Error('amendment 1 has wrong registration/implementation binding')
    # Prior art: XM1 amendment overlays (GRM, 2026). XM2 is a separate,
    # immutable descendant; never rewrite the earlier registration or pins.
    xm2 = xm2_registration()
    if xm2['xm1_amendment_sha256'] != sha(AMENDMENT):
        raise XM1Error('XM2 has wrong amendment 1 binding')
    for rel, digest in {**impl['pins'], **amendment['pins'], **xm2['pins']}.items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(ROOT/rel) != digest:
            raise XM1Error(f'implementation source drift: {rel}')
    return reg


@contextmanager
def legacy_environment(reg):
    # Prior art: GRM R1/LT1 pin-after-strip (GRM contributors, 2026).
    # A made-up GRM_LEGACY_DEFAULTS variable has no reader: pin real switches.
    env = {k: v for k, v in os.environ.items() if not k.startswith('GRM_')}
    env.update(reg['environment'])
    if 'GRM_QWEN35_FINAL_CHANNEL' in os.environ:
        env['GRM_QWEN35_FINAL_CHANNEL'] = os.environ['GRM_QWEN35_FINAL_CHANNEL']
    with patch.dict(os.environ, env, clear=True):
        yield


def normalize_value(text):
    return re.sub('[\u2010-\u2015\u2212]', '-', str(text)).casefold()


def value_score(answer, probe):
    from scripts.grm_rs1_read_strength_gpu import answer_verdict
    text = normalize_value(answer)
    expected = [normalize_value(v) for v in probe['expected_values']]
    rejected = [normalize_value(v) for v in probe['rejected_values']]
    hits = [v for v in expected if v in text]
    wrong = [v for v in rejected if v in text]
    refusal = bool(re.search(r"(?:don.t|do not) (?:have|know)|cannot (?:answer|provide)|unknown|not (?:available|provided)", text))
    return dict(**answer_verdict(answer, expected_values=probe['expected_values'],
                               rejected_values=probe['rejected_values']),
                refusal=refusal, refusal_rule='registered lexical diagnostic; raw served text authoritative')


def native_text(tokenizer, text):
    # Prior art: RS4 registered Harmony turns; model-native chat templates
    # (Hugging Face Transformers). Only markup changes, content is retained.
    if '<|start|>user<|message|>' in text:
        text = text.split('<|start|>user<|message|>', 1)[1]
    parts = text.split('<|end|><|start|>assistant<|channel|>final<|message|>', 1)
    messages = [{'role': 'user', 'content': parts[0].removesuffix('<|end|>')}]
    if len(parts) == 2:
        messages.append({'role': 'assistant', 'content': parts[1].removesuffix('<|end|>')})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


@dataclass
class Loaded:
    model: object
    tokenizer: object
    tc: object
    module: object
    attention_owner: object
    attention_name: str
    kind: str
    info: dict

    def attentions(self):
        for i, layer in enumerate(self.model.layers):
            att = getattr(layer, 'self_attn', None)
            if att is None and getattr(layer, 'is_attn', False):
                att = layer.mixer
            if att is not None:
                yield i, att

    def forward(self, ids, caches=None, offset=0):
        key = 'caches' if self.kind == 'partial_rope' else 'kv_caches'
        return self.model(np.array([ids], dtype=np.int64), **{key: caches},
                          position_offset=offset, last_token_only=True)


def gpu_loader(model_name, reg):
    # Prior art: the adapters' own from_pretrained and Trinity T1 driver (2026).
    from transformers import AutoTokenizer
    if model_name == 'gpt-oss':
        from core import gpt_oss20b_tc as mod
        from scripts.grm_det1_2_gpu import MODEL_DIR
        model, info = mod.GptOss20B_TC.from_pretrained(MODEL_DIR)
        path = MODEL_DIR
        owner, name = mod, 'sink_attention_tc'
    elif model_name == 'qwen35':
        from core import qwen35_tc as mod
        path = mod._snap()
        model, info = mod.Qwen35_TC.from_pretrained(path)
        owner, name = mod.F, 'scaled_dot_product_attention'
    elif model_name == 'minicpm3':
        from core import minicpm3_tc as mod
        path = mod._snap()
        model, info = mod.MiniCPM3_TC.from_pretrained(path)
        for layer in model.layers:
            layer.self_attn.absorbed_decode = False
        owner, name = mod.F, 'scaled_dot_product_attention'
    elif model_name == 'trinity':
        from scripts import trinity_nope_graft_width_sweep as mod
        path = mod.MODEL_DIR
        model, info, _, _ = mod.load_model(path)
        mod.install_arena_attention_hooks(model)
        model = mod.TrinityArenaModelAdapter(model)
        owner, name = mod, '_trinity_scaled_attention'
    else:
        raise XM1Error(f'no registered graft loader: {model_name}')
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    return Loaded(model, tok, mod.tc, mod, owner, name, reg['adapters'][model_name]['kind'], dict(info))


def capture(loaded, ids, shift):
    """Native adapter capture flags; exactly the kv_graft harvest protocol."""
    # Prior art: core.kv_graft.harvest_kv[_mla], GRM contributors (2026).
    ats = list(loaded.attentions())
    saved = [(a, getattr(a, 'live_shift', None), getattr(a, '_capture', False)) for _, a in ats]
    try:
        for _, att in ats:
            att.live_shift = shift
            att._capture = True
            att._captured = None
        with loaded.tc.no_grad():
            loaded.forward(ids)
        payload = {}
        for i, att in ats:
            if att._captured is None:
                raise XM1Error(f'native capture absent at layer {i}')
            payload[i] = tuple(np.array(x, copy=True) for x in att._captured)
        return payload
    finally:
        for att, shift0, cap0 in saved:
            att.live_shift, att._capture, att._captured = shift0, cap0, None


def seat(loaded, payload, width):
    """Relocate native keys by one constant delta, then use native injection."""
    # Prior art: ArenaCache._rs3_rotate_rows (GRM RS3, 2026), and Su et al.
    # RoFormer (2021), unverified — lead to check: arXiv 2104.09864.
    # Take constant-delta composition; specialize which payload carries RoPE.
    first = next(iter(payload.values()))
    n = first[0].shape[1 if loaded.kind == 'mla' else 2]
    delta = width - n
    if delta < 0:
        raise XM1Error('mount wider than registered native band')
    loaded.model.extend_rope(width + 4096)
    cos = loaded.model.rope_cos.numpy()[delta]
    sin = loaded.model.rope_sin.numpy()[delta]
    for i, att in loaded.attentions():
        a, b = (np.array(x, copy=True) for x in payload[i])
        rotary = b if loaded.kind == 'mla' else a
        nope = loaded.kind == 'nope_mixed' and not att.is_local_attention
        if not nope and delta:
            # The partial-RoPE table is narrower than the Qwen3.5 key.
            r = len(cos)
            h = r // 2
            head = rotary[..., :r]
            rotated = head * cos + np.concatenate([-head[..., h:], head[..., :h]], axis=-1) * sin
            rotary[..., :r] = rotated
        dtype = getattr(loaded.model, 'compute_dtype', 'bfloat16')
        if loaded.kind == 'nope_mixed':
            dtype = 'float32'
        ta, tb = (loaded.tc.tensor(np.ascontiguousarray(x)).astype(dtype) for x in (a, b))
        att.inject_kv = (ta, tb) if loaded.kind == 'mla' else (ta, tb, 1.0)
        att.graft_seats = n
        att.live_shift = width
    return dict(mount_ntok=n, mount_pos0=delta, live_shift=width,
                plan_head_last_position=width-1, delta_positions=delta,
                physical_mount_band=[0, n], n_sink=0, seat_near_live=True)


class Observer:
    """Read-only SDPA observer using the RS4 partition without modifying it."""
    def __init__(self, loaded, mount, fed, question):
        self.loaded, self.mount, self.fed, self.question = loaded, mount, fed, question
        self.rows = []
        self.step = []
        self.active = False

    def bounds(self, S):
        return live_band_bounds(S=S, n_sink=0, cur_mount_n=self.mount,
                                prior_ntok=0, fed_ntok=self.fed, question_ntok=self.question)

    def observe(self, q, k, **kw):
        # Prior art: core.grm_demand._full_mass and RS4 _full_split (2026).
        # Same last-query/heads reduction. Models without learned sinks get a
        # zero-probability sentinel column ONLY for RS4's host partition API.
        tc = self.loaded.tc
        L, S = q.shape[2], k.shape[2]
        scores = tc.matmul(q.slice(2, L-1, 1), k,
                           alpha=float(kw.get('scale') or q.shape[-1] ** -0.5), trans_b=True)
        mask = kw.get('attn_mask')
        if mask is not None:
            scores = scores + mask.slice(2, L-1, 1)
        probs = scores.softmax(-1).float().numpy()[0, :, 0, :]
        values = np.concatenate([probs, np.zeros((probs.shape[0], 1), np.float32)], axis=1)
        bounds = self.bounds(S)
        row = split_full_row(values, bounds)
        row['bounds'] = bounds
        row['layer_type'] = 'full_attention'
        ordinal = len(self.step)
        ats = list(self.loaded.attentions())
        if ordinal >= len(ats):
            raise XM1Error('extra attention call: layer inventory no longer matches')
        li, att = ats[ordinal]
        row['layer_index'] = li
        if getattr(att, 'is_local_attention', False):
            row['layer_type'] = 'sliding_attention'
        self.accept(row)

    def accept(self, row):
        b = row['bounds']
        widths = [b[n][1]-b[n][0] for n in ('sink_band','mount_band','prior_live_band','fed_text_band','question_band','answer_band')]
        if sum(widths) != b['S'] or not partition_sums_to_one(row) or not subbands_sum_to_live(row):
            raise XM1Error(f'RS4 partition failed: {row}')
        self.step.append(row)

    def __enter__(self):
        if self.loaded.kind == 'gqa_sink':
            # RS4's numerical observer methods are called VERBATIM, including
            # learned sinks and sliding-window intersections. Suppress inner
            # chunk calls just as the production observer does.
            from scripts.grm_rs4_ceiling_gpu import RowSplitMassObserver
            tap = RowSplitMassObserver.__new__(RowSplitMassObserver)
            tap._gpt = self.loaded.module
            tap._bounds_for = self.bounds
            self.sink_old = self.loaded.module.sink_attention_tc
            self.slide_old = self.loaded.module.sliding_sink_attention_tc
            self.inside_sliding = False
            def full(q,k,v,sinks,**kw):
                result=self.sink_old(q,k,v,sinks,**kw)
                if self.active and not self.inside_sliding:
                    row=tap._full_split(q,k,sinks,scale=kw['scale'],
                        attention_mask=kw.get('attention_mask'),num_heads_per_kv=kw.get('num_heads_per_kv',1))
                    self.accept(row)
                return result
            def sliding(q,k,v,sinks,**kw):
                self.inside_sliding=True
                try:
                    result=self.slide_old(q,k,v,sinks,**kw)
                finally:
                    self.inside_sliding=False
                if self.active:
                    row=tap._sliding_split(q,k,sinks,scale=kw['scale'], sliding_window=kw['sliding_window'],
                        num_heads_per_kv=kw.get('num_heads_per_kv',1),allowed_mask=kw.get('allowed_mask'))
                    self.accept(row)
                return result
            self.loaded.module.sink_attention_tc=full
            self.loaded.module.sliding_sink_attention_tc=sliding
            return self
        self.old = getattr(self.loaded.attention_owner, self.loaded.attention_name)
        def wrapper(q, k, v, *args, **kw):
            result = self.old(q, k, v, *args, **kw)
            if self.active:
                self.observe(q, k, **kw)
            return result
        setattr(self.loaded.attention_owner, self.loaded.attention_name, wrapper)
        return self

    def __exit__(self, *exc):
        if self.loaded.kind == 'gqa_sink':
            self.loaded.module.sink_attention_tc=self.sink_old
            self.loaded.module.sliding_sink_attention_tc=self.slide_old
            return
        setattr(self.loaded.attention_owner, self.loaded.attention_name, self.old)

    def forward(self, ids, caches, offset):
        self.step = []
        self.active = True
        try:
            result = self.loaded.forward(ids, caches, offset)
        finally:
            self.active = False
        if len(self.step) != len(list(self.loaded.attentions())):
            raise XM1Error('no/partial layer observation: not a valid cell')
        self.rows.append(self.step)
        return result


def native_cell(loaded, cell, reg):
    # Prior art: Qwen Team (2026), native thinking template; adapter and
    # source attribution in grm_xm1_qwen_final. All other models unchanged.
    if cell['model'] == 'qwen35':
        from scripts.grm_xm1_qwen_final import native_cell as final_cell
        return final_cell(loaded, cell, reg)
    return legacy_native_cell(loaded, cell, reg)


def legacy_native_cell(loaded, cell, reg):
    p = reg['probes'][cell['probe']]
    tok = loaded.tokenizer
    texts = p['capture_texts'] if cell['arm'] == 'C3l' else p['feed_texts']
    rendered = [native_text(tok, text) for text in texts]
    ids = [i for text in rendered for i in tok.encode(text, add_special_tokens=False)]
    question = tok.apply_chat_template([{'role':'user', 'content':p['question']}], tokenize=False, add_generation_prompt=True)
    qids = tok.encode(question, add_special_tokens=False)
    # Derive a single geometry for BOTH arms from both native tokenizations.
    counts = [sum(len(tok.encode(native_text(tok, t), add_special_tokens=False)) for t in p[key]) for key in ('capture_texts','feed_texts')]
    width = max(counts) + 96
    loaded.model.extend_rope(width + len(ids) + len(qids) + 4096)
    for _, att in loaded.attentions():
        att.inject_kv, att.graft_seats, att.live_shift = None, 0, width
        att.attention_mode = 'standard'
    caches, offset, mount, fed = None, 0, 0, 0
    seating = dict(seat_near_live=False, live_shift=width, n_sink=0, mount_ntok=0)
    with loaded.tc.no_grad():
        if cell['arm'] == 'C3l':
            payload = capture(loaded, ids, width)
            seating = seat(loaded, payload, width)
            mount = seating['mount_ntok']
        else:
            _, caches = loaded.forward(ids)
            fed, offset = len(ids), len(ids)
        generated = []
        with Observer(loaded, mount, fed, len(qids)) as observer:
            inputs = qids
            for step in range(32):
                logits, caches = observer.forward(inputs, caches, offset)
                offset += len(inputs)
                token = int(np.argmax(logits.numpy()[0, -1]))
                generated.append(token)
                for _, att in loaded.attentions():
                    att.inject_kv = None  # seats persist for cached decode
                eos = getattr(tok, 'eos_token_id', None)
                if token in (eos if isinstance(eos, (tuple, list)) else [eos]):
                    break
                inputs = [token]
    answer = tok.decode(generated, skip_special_tokens=True)
    by_type = {}
    for kind in ('full_attention', 'sliding_attention'):
        steps = [[r for r in step if r['layer_type']==kind] for step in observer.rows]
        by_type[kind] = mean_over_positions(steps)
    return dict(served=answer, value_span=value_score(answer, p), per_layer=by_type,
                observed_rows=observer.rows, seating=seating, source_texts=texts,
                rendered_texts=rendered, source_token_ids=ids, question_token_ids=qids,
                generated_token_ids=generated, model_info=loaded.info,
                protocol='native explicit capture/serve; no production routing; RS4 historical source-text contrast',
                recurrent_state_transferred=False if loaded.kind == 'partial_rope' else None)


def reference_diff(expected, observed):
    # Prior art: RS4 gate_g2, GRM contributors 2026. Exact float equality,
    # no epsilon, plus exact served text and correctness; missing is failure.
    keys = sorted(k for k in expected if '_mass_' in k)
    keys += ['served', 'correct', 'answer_positions']
    return {k: {'expected': expected[k], 'observed': observed.get(k)} for k in keys
            if expected[k] != observed.get(k)}


def reference_cell(cell, reg):
    """RS4's own run_arm, unchanged: full session state replay per cell."""
    from scripts import grm_rs4_ceiling_gpu as rs4
    # Prior art: Python tempfile.TemporaryDirectory lifetime management;
    # GRM RS4 (2026) repository scratch. Ours: worker-owned parent, disjoint
    # from pytest's disposable basetemp; numerical replay is unchanged.
    scratch = OUT/'worker_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='rs4_') as tmp:
        original = tempfile.mkdtemp
        def local_temp(*args, **kw):
            kw['dir'] = tmp
            return original(*args, **kw)
        with patch.object(rs4.tempfile, 'mkdtemp', local_temp):
            raw = rs4.run_arm(reg['probes'][cell['probe']]['session_id'], cell['arm'])
    table = rs4.assemble_table(raw['probes'])
    observed = next(r for r in table if r['probe_id']==cell['probe'] and r['variant']=='registered')
    expected = reg['rs4_references'][f"{cell['arm']}:{cell['probe']}"]
    diff = reference_diff(expected, observed)
    probe = next(r for r in raw['probes'] if r['probe_id']==cell['probe'] and r['variant']=='registered')
    return dict(reference_row=observed, reference_diff=diff, raw_rs4=raw,
                served=observed['served'], value_span=value_score(observed['served'], reg['probes'][cell['probe']]),
                per_layer=probe['mass']['by_layer_type'],
                parity_barrier='FAIL' if diff else 'PASS')


def cell_path(directory, cell, mode):
    return Path(directory)/'cells'/mode/cell['model']/f"{cell['arm']}__{cell['probe']}.json"


def binding(cell, mode):
    result = dict(cell=cell, mode=mode, registration_sha256=sha(REG),
                implementation_sha256=sha(IMPL), amendment_sha256=sha(AMENDMENT))
    if cell['model'] == 'qwen35':
        from scripts.grm_xm1_qwen_final import enabled
        if enabled():
            result['xm2_registration_sha256'] = sha(XM2_REG)
    return result


def verify_receipt(path, cell, mode):
    r = read(path)
    if r['binding'] != binding(cell, mode):
        raise XM1Error(f'resume binding mismatch: {path}')
    return r


def barrier(directory, reg):
    missing, differences = [], {}
    for cell in reg['cells']:
        if cell['model'] != 'gpt-oss':
            continue
        path = cell_path(directory, cell, 'gpu')
        if not path.exists():
            missing.append(str(path.relative_to(directory)))
            continue
        try:
            r = verify_receipt(path, cell, 'gpu')
        except XM1Error as exc:
            differences[path.name] = dict(status='BINDING_MISMATCH', error=str(exc))
            continue
        expected = reg['rs4_references'][f"{cell['arm']}:{cell['probe']}"]
        diff = reference_diff(expected, r.get('result', {}).get('reference_row', {}))
        if r['status'] != 'PASS' or diff:
            differences[path.name] = dict(status=r['status'], diff=diff)
    return dict(status='FAIL' if differences else 'BLOCKED' if missing else 'PASS', missing=missing, differences=differences)


def reference_progress(directory, cell, reg):
    # Prior art: RS4 exact comparison and LT1 create-only receipts (GRM, 2026).
    # Partial/failed reference progress is evidence, not a GPT scheduling gate.
    create(Path(directory)/'barriers'/'gpu'/f'{time.time_ns()}.json',
           dict(binding=binding(cell, 'gpu'), barrier=barrier(directory, reg)))


def device_memory():
    # Prior art: NVIDIA nvidia-smi device telemetry. Snapshot only, NOT peak
    # allocator usage. Unverified — lead to check: nvidia-smi memory.used.
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free',
                        '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=5)
    return dict(kind='device-wide snapshot MiB; not per-process/peak', returncode=p.returncode,
                stdout=p.stdout.strip(), stderr=p.stderr.strip())


def execute(cell, reg, directory=OUT, mode='cpu', loader=None):
    path = cell_path(directory, cell, mode)
    # Prior art: GRM XM1/RS4 reference barrier (2026). Amendment 1: gate
    # dependent models before both fresh execution and resume, never GPT-OSS.
    if mode == 'gpu' and cell['model'] != 'gpt-oss':
        gate = barrier(directory, reg)
        if gate['status'] != 'PASS':
            raise XM1Error('STOP: GPT-OSS RS4 parity barrier '+json.dumps(gate))
    if path.exists():
        result = verify_receipt(path, cell, mode)
        if mode == 'gpu' and cell['model'] == 'gpt-oss':
            reference_progress(directory, cell, reg)
        return result
    start = time.monotonic()
    result = dict(binding=binding(cell, mode), status='BLOCKED', evidence_class='CPU loader double' if mode=='cpu' else 'GPU end-to-end cell',
                  memory={'before':None, 'after':None, 'peak':None}, gpu_executed=False)
    if reg['adapters'][cell['model']]['status'] != 'READY':
        result.update(status='NO_GRAFT_PATH', reason=reg['adapters'][cell['model']])
    else:
        loaded = None
        try:
            with legacy_environment(reg):
                if mode == 'gpu':
                    result['memory']['before'] = device_memory()
                    result['gpu_executed'] = True
                if cell['model']=='gpt-oss' and mode=='gpu':
                    result['result'] = reference_cell(cell, reg)
                    result['status'] = result['result']['parity_barrier']
                else:
                    if loader is None:
                        loader = gpu_loader
                    loaded = loader(cell['model'], reg)
                    result['result'] = native_cell(loaded, cell, reg)
                    result['status'] = 'PASS'
                    if result['result'].get('decode_status', 'FINAL') != 'FINAL':
                        result['status'] = result['result']['decode_status']
                    if cell['model']=='gpt-oss':
                        result['reference_limit'] = 'Native CPU protocol double only; RS4 production replay is NOT exercised. GPU reference barrier remains BLOCKED.'
        except (Exception, TimeoutError) as exc:
            message = str(exc)
            oom = bool(re.search('out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation', message, re.I))
            result.update(status='NON_FIT' if mode=='gpu' and oom else 'ERROR', error_type=type(exc).__name__, error=message)
        finally:
            if loaded is not None:
                for _, att in loaded.attentions():
                    att.inject_kv = None
                    att.graft_seats = 0
            if mode=='gpu' and result['gpu_executed']:
                try:
                    result['memory']['after'] = device_memory()
                except Exception as exc:
                    result['memory']['after'] = {'error':str(exc)}
    result['elapsed_s'] = time.monotonic()-start
    create(path, result)
    if mode == 'gpu' and cell['model'] == 'gpt-oss':
        reference_progress(directory, cell, reg)
    return result


def dry_run(cell, reg):
    return dict(status='DRY_RUN', cell=cell, adapter=reg['adapters'][cell['model']],
                barrier_required=cell['model']!='gpt-oss', worker_cap_s=285, outer_cap_s=590,
                gpu_executed=False, non_fit_candidates=reg['non_fit_candidates'],
                source_texts_equal=reg['probes'][cell['probe']]['capture_texts']==reg['probes'][cell['probe']]['feed_texts'],
                estimate_s=cell['estimate_s'], budget_s=1800)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cpu-double', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(argv)
    reg = validate_registration()
    cell = next((c for c in reg['cells'] if all(c[k]==getattr(args,k) for k in ('model','arm','probe'))), None)
    if cell is None:
        parser.error('unregistered cell')
    if args.dry_run:
        print(json.dumps(dry_run(cell,reg), ensure_ascii=False))
        return 0
    if args.cpu_double:
        from scripts.grm_xm1_cpu import cpu_loader
        result = execute(cell,reg,args.output, 'cpu',cpu_loader)
    else:
        # Prior art: GRM CMC1 fail-fast flock lease (2026); never wait for or
        # signal someone else's process. Worst-case lost workers are charged
        # their entire reservation. Every attempt is immutable and receipted.
        from scripts.grm_cmc1_gpu_arms import gpu_lease
        target = cell_path(args.output,cell,'gpu')
        if target.exists():
            result = execute(cell,reg,args.output,'gpu')
        else:
            with gpu_lease(285,0):
                attempts = args.output/'attempts'/cell['model']
                used = 0.0
                for p in sorted(attempts.glob('*.json')):
                    attempt=read(p)
                    cp=cell_path(args.output,attempt['cell'],'gpu')
                    used += read(cp)['elapsed_s'] if cp.exists() else attempt['reservation_s']
                if used+285 > reg['gpu_budget_s_per_model']:
                    raise XM1Error(f'STOP: model budget cannot reserve another 285 s ({used=})')
                create(attempts/f'{time.time_ns()}.json',dict(cell=cell,reservation_s=285))
                result=execute(cell,reg,args.output,'gpu')
    print(json.dumps({'status':result['status'],'cell':cell}, ensure_ascii=False))
    return 0 if result['status'] in ('PASS','NO_GRAFT_PATH','NON_FIT') else 2

if __name__=='__main__':
    raise SystemExit(main())
