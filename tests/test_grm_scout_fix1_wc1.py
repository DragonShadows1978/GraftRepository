"""Prior art: WC1's deduplicated turn means (project contributors, 2026).
New fixture distinguishes graft count, measured token sum and missing evidence.
"""
import json
import pytest
from scripts import grm_wc1_results as wc1


def write_session(path, rows):
    path.mkdir()
    (path/'instrumentation.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return path


def row(turn, fitted, tokens):
    return dict(turn=turn, info={'mount_fitted': fitted},
                route_receipt={'fit': {'cur_mount_n': tokens}})


def test_wc1_token_seats_and_resume_dedup(tmp_path):
    first = write_session(tmp_path/'first', [row(1, [0, 1], 70), row(2, [], 0)])
    resumed = write_session(tmp_path/'resume', [row(1, [0], 900), row(3, [3], 20)])
    result = wc1._turn_stats([first, resumed])
    assert result['turns_counted'] == 3
    assert result['mean_resident_seats_per_turn'] == 1.0  # old graft-count key
    assert result['mean_resident_token_seats_per_turn'] == 30.0
    assert result['turns_with_token_seat_measurement'] == 3


def test_wc1_missing_token_count_is_unknown_not_zero_or_partial_mean(tmp_path):
    session = write_session(tmp_path/'s', [row(1, [0], 70), row(2, [1], None)])
    result = wc1._turn_stats([session])
    assert result['mean_resident_seats_per_turn'] == 1.0
    assert result['mean_resident_token_seats_per_turn'] is None
    assert result['turns_with_token_seat_measurement'] == 1


def test_wc1_nested_receipt_fallback(tmp_path):
    r = row(1, [4], 52)
    r['info']['route_receipt_record'] = r.pop('route_receipt')
    result = wc1._turn_stats([write_session(tmp_path/'s', [r])])
    assert result['mean_resident_token_seats_per_turn'] == 52.


def test_wc1_print_labels_both_columns(tmp_path, monkeypatch, capsys):
    entry = dict(score='1/1', all_probes='1/1', vs_96='reference',
                 mean_resident_seats_per_turn=2., mean_resident_token_seats_per_turn=70.)
    payload = dict(table=[{'width':96}], detail={'96':dict(census=entry)}, predictions=[])
    monkeypatch.setattr(wc1, 'build_results', lambda: payload)
    monkeypatch.setattr(wc1, 'ARTIFACT_DIR', tmp_path)
    monkeypatch.setattr(wc1, 'RESULTS', tmp_path/'result.json')
    wc1.main(['--print-table'])
    output = capsys.readouterr().out
    assert 'grafts' in output and 'token_seats' in output
    assert '70.0' in output
