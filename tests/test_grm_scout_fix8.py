"""Prior art: SC1.1 glyph tests and C7 CPU doubles (GRM, 2026), Unicode
Consortium UCD17/UAX15 (2025). Reuse exact binding and negative controls;
ours is the cross-channel Unicode boundary matrix. No new matching method.
"""
import json
from pathlib import Path
import pytest
from core.graft_arena import ArenaCache


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) flipped the shipped admission rule to `margin_first`
# and turned F1 / F2 / F5 / A1 ON.  F1 changes which nodes a digest leaves active, which is the input to this suite's inherited-key assertions.
#
# THIS SUITE'S ASSERTIONS ARE UNCHANGED -- they are the receipt for the
# pre-D2 behaviour this suite was written to pin, and `GRM_LEGACY_DEFAULTS=1`
# restores exactly that world.  The flipped behaviours have their own suites
# (tests/test_grm_f1_fold_retain.py, test_grm_f2_alias_guard.py,
# test_grm_f5_sole_binder_insurance.py, test_grm_a1_alias_fold.py).
@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    from core import grm_legacy_defaults as legacy_defaults
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")

from core.grm_admission import normalized_words, ordered_identifier_tokens, is_identifier_binding, split_member_binds_own_text
from core.grm_text_norm import normalize_glyphs

# Independent official UCD17 PropList Dash ranges, not the implementation's
# constant; includes U+10D6E (newer than this Python's Unicode tables).
DASHES = [0x2D,0x58A,0x5BE,0x1400,0x1806,*range(0x2010,0x2016),0x2053,
          0x207B,0x208B,0x2212,0x2E17,0x2E1A,0x2E3A,0x2E3B,0x2E40,
          0x2E5D,0x301C,0x3030,0x30A0,0xFE31,0xFE32,0xFE58,0xFE63,
          0xFF0D,0x10D6E,0x10EAD]


@pytest.mark.parametrize('cp',DASHES,ids=lambda cp:f'U+{cp:04X}')
def test_every_dash_cross_channel(cp,monkeypatch):
    monkeypatch.setenv('GRM_LSR_FIXES','1')
    s='C7-Fresh-0'.replace('-',chr(cp))
    assert normalize_glyphs(s)=='C7-Fresh-0'
    assert normalized_words(s)==['c7-fresh-0']
    assert ArenaCache._rare_tokens(s)=={'c7-fresh-0'}
    assert ArenaCache._query_content_tokens(s)=={'c7-fresh-0'}
    assert ArenaCache._node_text_tokens(s)=={'c7-fresh-0'}
    for question,candidate in [('C7-Fresh-0',s),(s,'C7-Fresh-0')]:
        ordered,rare=ordered_identifier_tokens(ArenaCache,question)
        kw=dict(ordered_identifier_tokens=ordered,rare_identifier_tokens=rare)
        assert is_identifier_binding(candidate_text=candidate,**kw)
        assert split_member_binds_own_text(text=candidate,**kw)
        assert not is_identifier_binding(candidate_text='C7-Fresh-9',**kw)
        assert not is_identifier_binding(candidate_text='C7 Fresh 0',**kw)


@pytest.mark.parametrize('s',['Ｃ７－Ｆｒｅｓｈ－０','C⁷‑Fresh‑₀','**C7‑Fresh‑0**'])
def test_nfkc_and_existing_emphasis(s):
    assert normalized_words(s)==['c7-fresh-0']
    assert normalize_glyphs(normalize_glyphs(s))==normalize_glyphs(s)
    assert ArenaCache._node_text_tokens(s)=={'c7-fresh-0'}


def test_unicode_casefold_and_case_bearing_caps():
    s='C7-Straße-0'
    assert normalized_words(s)==['c7-strasse-0']
    assert ArenaCache._rare_tokens(s)=={'c7-strasse-0'}
    assert ArenaCache._query_content_tokens(s)=={'c7-strasse-0'}
    assert ArenaCache._node_text_tokens(s)=={'c7-strasse-0'}
    assert normalize_glyphs('Harbor')=='Harbor'
    assert 'harbor' in ArenaCache._caps_tokens('Harbor',False)


def test_digest_own_identifiers_and_inherited_keys(tmp_path,monkeypatch):
    from scripts.grm_c7_diagnose import repository
    repo=repository(tmp_path/'repo',monkeypatch);a=repo.arena
    try:
        i=a.feed('The current C7-Fresh-0 value is Basalt-811.')
        j,_=a._deposit_consolidation([i],'The current Ｃ７‑Fresh‑1 value is Basalt‑812.')
        assert a.grafts[i]['retired']
        assert {'c7-fresh-0','c7-fresh-1'}<=a.grafts[j]['rare']
        for token,want in [('c7-fresh-1',True),('c7-fresh-0',False)]:
            assert is_identifier_binding(candidate_text=a.grafts[j]['text'],
                ordered_identifier_tokens=[token],rare_identifier_tokens=[token]) is want
    finally:repo.close()


def test_receipt_bound_red_green_and_c2():
    root=Path(__file__).resolve().parents[1]/'artifacts/grm_scout_fix8'
    red=json.loads((root/'red.json').read_text());green=json.loads((root/'green.json').read_text())
    assert red['bind_count']==red['generated_count']==0
    assert green['bind_count']==green['generated_count']==24
    assert [r['probe_id'] for r in red['rows']]==[r['probe_id'] for r in green['rows']]
    assert all(r['refused'] and not r['mounted_ids'] for r in red['rows'])
    assert all(not r['refused'] and r['mounted_ids'] for r in green['rows'])
    assert red['c2_identical']==green['c2_identical']==132
    assert red['c2']==green['c2']
