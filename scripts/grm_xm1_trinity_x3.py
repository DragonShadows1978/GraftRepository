#!/usr/bin/env python3
"""Opt-in Trinity matched-token diagnostic; OFF is the immutable XM1 worker.

Prior art: GRM contributors (2026), XM1 native_cell/capture/seat/Observer,
RS3 capture pin and near-live seating, RS4 partitions, LT1 immutable pins and
CMC1 worker leases. Reused verbatim where possible. Su et al. (2021), RoFormer
rotation composition, unverified — lead to check 2104.09864. Ours: isolate the
same-token/position contrast and label decode steps separately from answers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import grm_xm1_parity as xm
from scripts.grm_xm1_registration import read, create, sha

OUT = ROOT / 'artifacts/grm_xm3'
REG = OUT / 'registration.json'
PINS = OUT / 'implementation_amendment_3.json'


def validate():
    reg = read(REG)
    amendment = read(PINS)
    if amendment['previous_amendment_2_sha256'] != sha(OUT / 'implementation_amendment_2.json'):
        raise xm.XM1Error('X3 second amendment drift')
    if amendment['previous_amendment_sha256'] != sha(OUT / 'implementation_amendment_1.json'):
        raise xm.XM1Error('X3 previous amendment drift')
    if amendment['previous_implementation_sha256'] != sha(OUT / 'implementation_pins.json'):
        raise xm.XM1Error('X3 implementation chain drift')
    if amendment['registration_sha256'] != sha(REG):
        raise xm.XM1Error('X3 registration binding drift')
    for rel, digest in {**reg['pins'], **amendment['pins']}.items():
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or sha(ROOT / p) != digest:
            raise xm.XM1Error(f'X3 source drift: {rel}')
    return reg


def payload_digest(payload):
    # Prior art: XM1/LT1 content binding (GRM 2026); tensor shape/dtype included.
    h = hashlib.sha256()
    for li, arrays in sorted(payload.items()):
        h.update(str(li).encode())
        for a in arrays:
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes(order='C'))
    return h.hexdigest()


def matched_cell(loaded, cell, base, *, enabled=False, capture_pin='off', near=False):
    # Prior art: XM1 native_cell (GRM 2026). OFF delegates without mutation.
    if not enabled:
        return xm.native_cell(loaded, cell, base)
    if loaded.kind != 'nope_mixed' or cell['model'] != 'trinity':
        raise xm.XM1Error('X3 is Trinity-only')
    if capture_pin not in ('off', 'live'):
        raise xm.XM1Error('unknown capture pin')
    p = base['probes'][cell['probe']]
    tok = loaded.tokenizer
    texts = p['capture_texts']
    rendered = [xm.native_text(tok, t) for t in texts]
    ids = [i for t in rendered for i in tok.encode(t, add_special_tokens=False)]
    question = tok.apply_chat_template([{'role': 'user', 'content': p['question']}],
                                      tokenize=False, add_generation_prompt=True)
    qids = tok.encode(question, add_special_tokens=False)
    n = len(ids)
    if not 0 < n <= 96 or n + len(qids) + 32 >= 2048:
        raise xm.XM1Error('STOP: registered 96-seat/window fit rail')
    width = max(96, n)
    live_text_start = width - n
    loaded.model.extend_rope(width + len(qids) + 64)
    for _, att in loaded.attentions():
        att.inject_kv, att.graft_seats, att.live_shift = None, 0, width
        att.attention_mode = 'standard'
    caches, mount, fed, offset = None, 0, 0, 0
    seating = dict(n_sink=0, live_shift=width, live_text_start=live_text_start,
                   question_pos0=width, mount_ntok=0, seat_near_live=near,
                   capture_pin=capture_pin)
    with loaded.tc.no_grad():
        if cell['arm'] == 'C3l':
            shift = live_text_start if capture_pin == 'live' else 0
            payload = xm.capture(loaded, ids, shift)
            digest = payload_digest(payload)
            # Prior art: RS3/XM1 constant-delta composition (GRM 2026),
            # Su et al. 2021. Near OFF seats at 0 using existing seat(width=n).
            positioned = xm.seat(loaded, payload, width if near else n)
            for _, att in loaded.attentions():
                att.live_shift = width
            seating.update(positioned)
            seating.update(live_shift=width, seat_near_live=near,
                           capture_shift=shift, graft_sha256=digest,
                           payload_unchanged=payload_digest(payload) == digest)
            mount = n
        elif cell['arm'] == 'C5':
            # Align C5 text to the ON mount's logical positions; then reset
            # offset to 0 with shift=width so the question positions coincide.
            for _, att in loaded.attentions():
                att.live_shift = live_text_start
            _, caches = loaded.forward(ids)
            for _, att in loaded.attentions():
                att.live_shift = width
            fed = n
        else:
            raise xm.XM1Error('unregistered native arm')
        generated = []
        with xm.Observer(loaded, mount, fed, len(qids)) as observer:
            inputs = qids
            for step in range(32):
                logits, caches = observer.forward(inputs, caches, offset)
                offset += len(inputs)
                token = int(np.argmax(logits.numpy()[0, -1]))
                generated.append(token)
                for _, att in loaded.attentions():
                    att.inject_kv = None
                eos = getattr(tok, 'eos_token_id', None)
                if token in (eos if isinstance(eos, (tuple, list)) else [eos]):
                    break
                inputs = [token]
    served = tok.decode(generated, skip_special_tokens=True)
    score = xm.value_score(served, p)
    by_type = {kind: xm.mean_over_positions([[r for r in step if r['layer_type'] == kind]
                for step in observer.rows]) for kind in ('full_attention', 'sliding_attention')}
    return dict(served=served, raw_generated_text=tok.decode(generated, skip_special_tokens=False),
                value_span=score, generated_token_ids=generated, per_layer=by_type,
                observed_rows=observer.rows, seating=seating, source_texts=texts,
                rendered_texts=rendered, source_token_ids=ids, question_token_ids=qids,
                rendered_question=question, model_info=loaded.info,
                decode_window=dict(steps=len(generated), first_query='last question token predicts first output',
                    subsequent_queries='preceding generated token predicts next output',
                    semantic_answer_valid=bool(score['correct']),
                    mean_label='decode-position mean; answer quality gated separately'),
                protocol='X3 matched source IDs and near-live positions; explicit native adapter, not production replay')


def cell_binding(probe, treatment):
    return dict(registration_sha256=sha(REG), implementation_sha256=sha(PINS),
                probe=probe, treatment=treatment)


def prerequisites(reg, probe, treatment, output=OUT):
    # Prior art: LT1 fail-closed prerequisites + CMC1/XM1 reservations (2026).
    # Charge whole reservations, even failed workers. No budget refunds.
    attempts = list((output / 'attempts').glob('*.json'))
    used = sum(read(p)['reservation_s'] for p in attempts)
    if used + reg['gpu']['reservation_per_cell_s'] > reg['gpu']['total_budget_s']:
        raise xm.XM1Error('STOP: GPU budget exhausted')
    if attempts:
        last = max(read(p)['reserved_at'] for p in attempts)
        if time.time() < last + reg['gpu']['reservation_per_cell_s'] + 30:
            raise xm.XM1Error('STOP: 30 s inter-lease gap not yet available; no waiting')
    if treatment in ('C0', 'C1l', 'C2', 'C3l'):
        path = output / 'cells' / f'C5__{probe}.json'
        if not path.exists():
            raise xm.XM1Error('STOP: matched C5 control missing')
        control = read(path)
        if (control['binding'] != cell_binding(probe, 'C5') or control['status'] != 'PASS'
                or not control['result']['value_span']['correct']):
            raise xm.XM1Error('STOP RED: matched C5 control has no correct answer')


def parity_comparison(control, mounted):
    # Prior art: XM1 registered fed-minus-mount falsifier (GRM 2026).
    # Reuse threshold 0.10 unchanged; a correct answer alone is not clearance.
    gaps = {}
    for kind in ('full_attention', 'sliding_attention'):
        f = control['per_layer'][kind]['mean_over_answer_positions']['fed_text_mass']
        m = mounted['per_layer'][kind]['mean_over_answer_positions']['mounted_mass']
        if f is None or m is None or not np.isfinite([f, m]).all():
            raise xm.XM1Error('STOP: missing/nonfinite parity mass')
        gaps[kind] = f - m
    correct = bool(control['value_span']['correct'] and mounted['value_span']['correct'])
    return dict(fed_minus_mount=gaps, threshold=0.10,
                status='PASS' if correct and gaps['full_attention'] <= 0.10 else 'RED_PARITY',
                both_correct=correct, evidence_limit='free decode trajectories; no teacher-forced parity claim')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--treatment', required=True)
    parser.add_argument('--xm3-matched', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    reg = validate()
    if args.probe not in reg['probes'] or args.treatment not in reg['arms']:
        parser.error('unregistered X3 cell')
    arm = reg['arms'][args.treatment]
    if args.xm3_matched != arm['matched']:
        parser.error('--xm3-matched must match registered treatment')
    base = xm.validate_registration()
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN', gpu_executed=False, probe=args.probe,
                             treatment=args.treatment, geometry=reg['geometry'],
                             reservation_s=reg['gpu']['reservation_per_cell_s'],
                             control_gate=reg['gpu']['control_gate'])))
        return 0
    target = OUT / 'cells' / f'{args.treatment}__{args.probe}.json'
    binding = cell_binding(args.probe, args.treatment)
    if target.exists():
        old = read(target)
        if old['binding'] != binding:
            raise xm.XM1Error('STOP: resume binding mismatch')
        print(json.dumps({'status': old['status'], 'resumed': True}))
        return 0 if old['status'] == 'PASS' else 2
    if xm.barrier(ROOT / 'artifacts/grm_xm1/amendment_1_run', base)['status'] != 'PASS':
        raise xm.XM1Error('STOP: historical GPT-OSS reference barrier')
    # Validate external live sources against repo-relative snapshots before GPU.
    for rel, source in reg['external_sources'].items():
        if sha(Path(source['read_only_origin'])) != sha(ROOT / rel):
            raise xm.XM1Error(f'STOP: external source drift for {rel}')
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    with gpu_lease(reg['gpu']['reservation_per_cell_s'], 0):
        prerequisites(reg, args.probe, args.treatment)
        create(OUT / 'attempts' / f'{time.time_ns()}.json',
               dict(binding=binding, reservation_s=reg['gpu']['reservation_per_cell_s'], reserved_at=time.time()))
        receipt = dict(binding=binding, status='ERROR', gpu_executed=True,
                       evidence_class='GPU adapter diagnostic; no production certification')
        started = time.monotonic()
        try:
            with xm.legacy_environment(base):
                loaded = xm.gpu_loader('trinity', base)
                result = matched_cell(loaded, dict(model='trinity', probe=args.probe, arm=arm['arm']), base,
                                      enabled=args.xm3_matched, capture_pin=arm.get('pin', 'off'),
                                      near=arm.get('near', False))
                receipt['result'] = result
                receipt['status'] = 'PASS' if result['value_span']['correct'] else 'RED_NO_ANSWER'
                if args.treatment in ('C0', 'C1l', 'C2', 'C3l'):
                    control = read(OUT / 'cells' / f'C5__{args.probe}.json')['result']
                    receipt['comparison'] = parity_comparison(control, result)
                    if receipt['comparison']['status'] != 'PASS':
                        receipt['status'] = 'RED_PARITY'
        except Exception as exc:
            receipt.update(error_type=type(exc).__name__, error=str(exc))
        finally:
            receipt['elapsed_s'] = time.monotonic() - started
            create(target, receipt)
    print(json.dumps({'status': receipt['status'], 'path': str(target.relative_to(ROOT))}))
    return 0 if receipt['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
