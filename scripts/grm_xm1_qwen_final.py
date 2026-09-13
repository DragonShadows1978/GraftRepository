"""XM2 Qwen final-channel adapter, CPU-importable; no model import.

Prior art: Qwen Team (2026), Qwen3.5-9B chat_template.jinja, revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a, copied to artifacts/grm_xm2/sources.
Upstream unverified — lead to check: Qwen3.5-9B enable_thinking chat template.
Taken: enable_thinking=False emits a closed empty thinking block. Ours:
explicit final input-query token indices, raw trace retention, bounded decode.
GRM contributors (2026) XM1 capture/seat and RS4 mean reused unchanged.
"""
import hashlib
import os
import numpy as np


def enabled():
    value = os.environ.get('GRM_QWEN35_FINAL_CHANNEL', '1').strip().lower()
    if value in ('1', 'true', 'on', 'yes'):
        return True
    if value in ('0', 'false', 'off', 'no'):
        return False
    raise ValueError('invalid GRM_QWEN35_FINAL_CHANNEL: ' + value)


def final_indices(tok, generated, prompt):
    """Indices of content tokens, excluding thought, control and leading space.

    Prior art: Qwen Team (2026) think delimiters in the pinned chat template.
    Ours: token-index state machine; each content token i maps to observed
    input-query row i+1, not row i that *predicted* it. A missing close in an
    open-thinking prompt yields no answer, never leaked reasoning.
    """
    opening = tok.encode('<think>', add_special_tokens=False)
    closing = tok.encode('</think>', add_special_tokens=False)
    if len(opening) != 1 or len(closing) != 1:
        raise ValueError('Qwen thinking delimiters must each be one token')
    eos = tok.eos_token_id
    stops = set(eos if isinstance(eos, (tuple, list)) else [eos])
    special = set(getattr(tok, 'all_special_ids', []))
    thinking = prompt.rfind('<think>') > prompt.rfind('</think>')
    out = []
    for i, token in enumerate(generated):
        if token in stops:
            break
        if token == opening[0]:
            thinking = True
        elif token == closing[0]:
            thinking = False
        elif not thinking and token not in special:
            if out or tok.decode([token], skip_special_tokens=True).strip():
                out.append(i)
    return out


def native_cell(loaded, cell, reg):
    from scripts import grm_xm1_parity as xm
    # Prior art: XM1 (GRM, 2026). OFF directly calls the preserved legacy body;
    # no template kwargs, decode limits, extra forwards or receipt fields.
    if not enabled():
        return xm.legacy_native_cell(loaded, cell, reg)
    p, tok = reg['probes'][cell['probe']], loaded.tokenizer
    texts = p['capture_texts'] if cell['arm'] == 'C3l' else p['feed_texts']
    rendered = [xm.native_text(tok, text) for text in texts]
    ids = [i for text in rendered for i in tok.encode(text, add_special_tokens=False)]
    question = tok.apply_chat_template(
        [{'role': 'user', 'content': p['question']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    qids = tok.encode(question, add_special_tokens=False)
    counts = [sum(len(tok.encode(xm.native_text(tok, t), add_special_tokens=False))
                  for t in p[key]) for key in ('capture_texts', 'feed_texts')]
    width = max(counts) + 96
    loaded.model.extend_rope(width + len(ids) + len(qids) + 4096)
    for _, att in loaded.attentions():
        att.inject_kv, att.graft_seats, att.live_shift = None, 0, width
        att.attention_mode = 'standard'
    caches, offset, mount, fed = None, 0, 0, 0
    seating = dict(seat_near_live=False, live_shift=width, n_sink=0, mount_ntok=0)
    capture_info = None
    with loaded.tc.no_grad():
        if cell['arm'] == 'C3l':
            pin = cell.get('capture_pin', 'live')
            near = cell.get('seat_near_live', True)
            if pin not in ('off', 'live') or not isinstance(near, bool):
                raise ValueError('unregistered Qwen capture/seat geometry')
            # Prior art: RS3 (GRM, 2026), controlled capture + constant-delta
            # relocation. OFF here is fresh cleared native geometry (0), NOT
            # arbitrary residual state of a production arena.
            shift = width if pin == 'live' else 0
            payload = xm.capture(loaded, ids, shift)
            capture_info = dict(capture_pin=pin, capture_shift=shift,
                capture_ntok=len(ids), capture_texts=texts,
                payload_sha256={str(i): hashlib.sha256(b''.join(
                    np.ascontiguousarray(x).tobytes() for x in pair)).hexdigest()
                    for i, pair in payload.items()})
            seating = xm.seat(loaded, payload, width if near else len(ids))
            for _, att in loaded.attentions():
                att.live_shift = width
            seating.update(live_shift=width, seat_near_live=near)
            mount = seating['mount_ntok']
        else:
            _, caches = loaded.forward(ids)
            fed, offset = len(ids), len(ids)
        generated, ended = [], False
        with xm.Observer(loaded, mount, fed, len(qids)) as observer:
            inputs = qids
            for step in range(32):
                logits, caches = observer.forward(inputs, caches, offset)
                offset += len(inputs)
                token = int(np.argmax(logits.numpy()[0, -1]))
                generated.append(token)
                for _, att in loaded.attentions():
                    att.inject_kv = None
                eos = tok.eos_token_id
                if token in (eos if isinstance(eos, (tuple, list)) else [eos]):
                    ended = True
                    break
                inputs = [token]
            if not ended:
                # Observe the last generated input query on a truncated trace;
                # do not sample another output or count the final EOS row.
                observer.forward([generated[-1]], caches, offset)
    selected = final_indices(tok, generated, question)
    row_indices = [i + 1 for i in selected]
    final_rows = [observer.rows[i] for i in row_indices]
    answer = tok.decode([generated[i] for i in selected], skip_special_tokens=True)
    by_type = {kind: xm.mean_over_positions(
        [[r for r in step if r['layer_type'] == kind] for step in final_rows])
        for kind in ('full_attention', 'sliding_attention')}
    return dict(served=answer, value_span=xm.value_score(answer, p), per_layer=by_type,
        observed_rows=observer.rows, seating=seating, capture=capture_info,
        source_texts=texts, rendered_texts=rendered, source_token_ids=ids,
        question_token_ids=qids, question_rendered=question,
        generated_token_ids=generated, model_info=loaded.info,
        raw_served=tok.decode(generated, skip_special_tokens=False),
        final_channel=True, final_token_indices=selected,
        final_observation_indices=row_indices,
        answer_position_convention='input query for each final content token i: observation i+1; prefill/thinking/control excluded',
        decode_status='NO_FINAL' if not answer.strip() else 'FINAL' if ended else 'TRUNCATED_FINAL',
        protocol='XM2 native explicit capture/serve; historical source contrast; final input-query window',
        recurrent_state_transferred=False)
