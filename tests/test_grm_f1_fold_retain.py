"""GRM-F1 — source retention after a fold, flag OFF byte-identical.

What this pins, and what it deliberately does not:
  * OFF is the pre-F1 branch. Every OFF assertion is copied from the
    behaviour FIX-5's own suite already pins (sources RETIRED, digest is the
    only routable record), so a regression here is a regression there.
  * ON adds the digest instead of substituting it: sources stay ACTIVE and
    ROUTABLE, lineage is recorded both ways, and the librarian does not
    re-select the retained window forever.
  * SUPERSESSION is unchanged in BOTH states — a correction still retires
    the stale value and the digest carrying it. This is the assertion that
    matters most: retention is a compression policy, never a correctness
    one, and a retained source must never resurrect a corrected fact.
  * GRM-A1's lineage is invariant under the flag (the alias merge pins
    `retain=False`; a bare active alias edge IS the RD2 defect).

Prior art: the FIX-5 CPU fold harness and `scripts/grm_c7_diagnose`'s
numerical doubles (GRM contributors, 2026), reused verbatim as the stimulus
and the repository; the A1 suite's flag-OFF byte-identity shape
(`tests/test_grm_a1_alias_fold.py::test_flag_off_byte_identical`), reused as
the structure of the OFF proof. New here: the retention assertions and the
retained-source residency split. No prior art known to me for this exact
suite; unverified against the wider literature (no network in this sandbox)
— lead to check. CPU author evidence only: no GPU, no model-quality claim.
"""
import json
import os
from pathlib import Path
import re

import pytest

from core import grm_fold_retain as fr
from core import grm_legacy_defaults as legacy_defaults
from scripts.grm_c7_diagnose import Model, repository

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) flipped F1's shipped default from OFF to ON.  THIS
# SUITE'S ASSERTIONS ARE UNCHANGED: every one of them still pins the
# pre-D2 behaviour, which is exactly what ``GRM_LEGACY_DEFAULTS=1``
# selects, and pinning it here is what keeps this file a receipt for the
# OFF branch rather than a stale expectation of a default that moved.
#
# Two pins, because this suite resolves the flag two ways:
#   * ``_grm_d2_legacy_defaults`` pins the PROCESS environment, for every
#     test that goes through ``repository()`` / the arena;
#   * ``legacy()`` builds the explicit ``environ=`` dicts the resolver
#     tests pass directly, since those never consult ``os.environ``.
# Neither adds or removes an assertion.

def legacy(**extra):
    """An ``environ`` mapping pinned to the pre-D2 defaults."""
    return {legacy_defaults.ENV_NAME: "1", **extra}


@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")

#: Four turns whose facts are identifiers FIX-5's `_fact_set` will demand, so
#: the extractive digest below clears MIN_FOLD_KEEP and the fold is ACCEPTED
#: — the only branch in which the source lifecycle is decided at all.
SOURCES = [
    "The Breakwater beacon sits at grid BX-44 on the northern arc.",
    "Promenade lamp spacing was settled at 9 voxels end to end.",
    "The Commtower maintenance lead is Iona Vale as of this quarter.",
    "Hull-1 plating is the silver alloy, not the matte one.",
]


class EnumeratedModel(Model):
    """Extractive digest: echo the identifiers of the enumerated sources.

    Prior art: `tests/test_grm_scout_fix5.py::EnumeratedModel` (GRM
    contributors, 2026), reused verbatim. The stimulus reads ONLY the
    `[source N]` spans FIX-5 itself writes into the prompt — never the
    mounted text, never an expected answer, never `need_count`.
    """

    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            prompt = self.codec.decode(ids[0])
            lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
            tokens = sorted(set(re.findall(
                r'\b[A-Za-z][\w-]*\d[\w-]*\b', ' '.join(lines))))
            words = sorted(set(re.findall(
                r'\b[A-Z][a-z]+\b', ' '.join(lines))))
            self.fold_output = (
                'retained identifiers in the archival record include '
                + ', '.join(tokens + words) + '.<|end|>')
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)


def seeded(tmp_path, monkeypatch, *, retain=None, texts=SOURCES):
    repo = repository(tmp_path / 'repo', monkeypatch)
    if retain is not None:
        repo.arena.fold_retain_sources = bool(retain)
    repo.arena.m = EnumeratedModel(repo.arena.m.codec)
    ids = [repo.arena.deposit(t) for t in texts]
    return repo, ids


def fold(repo, ids):
    didx, text = repo.arena.consolidate(ids)
    assert didx is not None, repo.arena.last_consolidation_result
    return didx, text


def route_base(arena):
    """The routing candidate base — the ONE place `retired` decides
    reachability (`core/graft_arena.py::_route_cand_base`)."""
    return set(arena._route_cand_base())


# ------------------------------------------------------------------ the flag

def test_flag_default_off():
    assert fr.retain_sources_enabled() is False
    assert fr.retain_sources_enabled(environ=legacy()) is False
    assert fr.env_retain_sources_override(environ=legacy()) is None


@pytest.mark.parametrize('token', ['1', 'true', 'TRUE', 'yes', 'on', ' On '])
def test_flag_env_on(token):
    assert fr.retain_sources_enabled(
        environ=legacy(**{fr.ENV_NAME: token})) is True


@pytest.mark.parametrize('token', ['0', 'false', 'no', 'off', '', '  '])
def test_flag_env_off(token):
    assert fr.retain_sources_enabled(
        environ=legacy(**{fr.ENV_NAME: token})) is False


@pytest.mark.parametrize('token', ['maybe', '2', 'ON!', 'yes please', '-1'])
def test_flag_unknown_token_fails_closed(token):
    """A malformed operator escape never silently enables the behaviour —
    the A1 / L2 / A-DEC precedent."""
    assert fr.retain_sources_enabled(
        environ=legacy(**{fr.ENV_NAME: token})) is False
    # GRM-D2: an unknown token now returns None (decline to decide) rather
    # than False (pin OFF), so the caller falls through to the SHIPPED
    # default -- which under this suite's legacy pin is still OFF.  The
    # behaviour this test names ("fails closed") is unchanged; the rung it
    # falls closed ON moved from the override helper to the default.
    assert fr.env_retain_sources_override(
        environ=legacy(**{fr.ENV_NAME: token})) is None


def test_explicit_beats_env():
    env = legacy(**{fr.ENV_NAME: '1'})
    assert fr.retain_sources_enabled(False, environ=env) is False
    assert fr.retain_sources_enabled(
        True, environ=legacy(**{fr.ENV_NAME: '0'})) is True


def test_arena_resolves_flag_once_at_construction(tmp_path, monkeypatch):
    monkeypatch.setenv(fr.ENV_NAME, '1')
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        assert repo.arena.fold_retain_sources is True
        # A mid-session environment change cannot alter this conversation's
        # fold lifecycle — the same freeze A1/L2/A-DEC take.
        monkeypatch.delenv(fr.ENV_NAME)
        assert repo.arena.fold_retain_sources is True
    finally:
        repo.close()


def test_arena_default_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv(fr.ENV_NAME, raising=False)
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        assert repo.arena.fold_retain_sources is False
    finally:
        repo.close()


# ------------------------------------------------------- OFF byte-identical

def test_flag_off_byte_identical(tmp_path, monkeypatch):
    """OFF reproduces the pre-F1 fold EXACTLY: same digest text, same node
    table, same lifecycle, and no F1 key anywhere in the metadata."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=False)
    try:
        a = repo.arena
        didx, text = fold(repo, ids)
        # 1. Sources retired -> out of the routing candidate base.
        assert all(a.grafts[i]['retired'] for i in ids)
        base = route_base(a)
        assert not (set(ids) & base)
        assert didx in base
        # 2. The digest's own record is the pre-F1 one.
        assert a.grafts[didx]['sources'] == list(ids)
        assert a.grafts[didx]['kind'] == 'digest'
        # 3. No F1 lineage key is written ANYWHERE.
        for g in a.grafts:
            meta = g.get('metadata') or {}
            assert fr.LINEAGE_DIGEST_OF not in meta
            assert fr.LINEAGE_RETAINED_SOURCES not in meta
        # 4. No source gained the anti-reselect exemption.
        assert not any(a.grafts[i].get('no_fold') for i in ids)
        assert text
    finally:
        repo.close()


def test_off_fold_once_leaves_digest_metadata_untouched(tmp_path,
                                                        monkeypatch):
    """REGRESSION PIN, from a real OFF-arm break this seat introduced and
    measured against baseline core.

    An earlier F1 draft completed the DIGEST's metadata through
    `_ensure_lifecycle` unconditionally in `_fold_once`. That is not
    byte-identical with the flag OFF: it normalizes `metadata.active` on a
    digest the pre-F1 path left with no metadata at all, and
    `correct_memory` picks its targets with `meta.get("active", True)` — so
    an OFF-arm correction began superseding an extra, already-retired digest
    (measured: supersedes [0, 10] -> [0, 8, 10],
    artifacts/grm_f1/off_identity/). The lifecycle completion is now gated
    strictly on `retained`.

    What this pins: with the flag OFF, `_fold_once` adds NO metadata to the
    digest and marks nothing dirty.
    """
    repo, ids = seeded(tmp_path, monkeypatch, retain=False)
    try:
        a = repo.arena
        dirty_before = dict(repo.dirty_nodes)
        assert repo._fold_once([('digest', list(ids))]) is True
        didx = repo.fold_history[-1]['digest_idx']
        assert 'metadata' not in a.grafts[didx]
        assert didx not in set(repo.dirty_nodes) - set(dirty_before)
        # The fold event still RECORDS which lifecycle ran — an additive
        # receipt, consulted by no serving path.
        assert repo.fold_history[-1]['retained_sources'] == []
        assert repo.fold_history[-1]['fold_lineage'] == (
            fr.LINEAGE_SOURCES_RETIRED)
    finally:
        repo.close()


def test_off_correction_targets_only_active_nodes(tmp_path, monkeypatch):
    """The consequence the pin above protects: an OFF-arm correction must
    not reach a node the fold already retired."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=False)
    try:
        a = repo.arena
        assert repo._fold_once([('digest', list(ids))]) is True
        didx = repo.fold_history[-1]['digest_idx']
        new = repo.correct_memory(
            'BX-44', 'The Breakwater beacon sits at grid CX-97, corrected.')
        supersedes = a.grafts[new]['metadata']['supersedes']
        # The live digest is superseded; the fold-retired sources are NOT
        # (they were already inactive and invisible to correct_memory).
        assert didx in supersedes
        assert not set(ids) & set(supersedes)
    finally:
        repo.close()


def test_off_and_on_digests_are_identical_text(tmp_path, monkeypatch):
    """The DIGEST is the same node either way — retention changes only what
    happens to the sources, never what the fold produces."""
    off_repo, off_ids = seeded(tmp_path / 'off', monkeypatch, retain=False)
    on_repo, on_ids = seeded(tmp_path / 'on', monkeypatch, retain=True)
    try:
        off_idx, off_text = fold(off_repo, off_ids)
        on_idx, on_text = fold(on_repo, on_ids)
        assert off_text == on_text
        assert off_idx == on_idx
        assert (off_repo.arena.grafts[off_idx]['text']
                == on_repo.arena.grafts[on_idx]['text'])
        assert (off_repo.arena.grafts[off_idx]['sources']
                == on_repo.arena.grafts[on_idx]['sources'])
        assert (off_repo.arena.last_consolidation_result
                == on_repo.arena.last_consolidation_result)
    finally:
        off_repo.close()
        on_repo.close()


# ------------------------------------------------------------ ON: retention

def test_on_sources_stay_active_and_routable(tmp_path, monkeypatch):
    """The mission's measurement, at the unit scale: after a fold the source
    nodes are STILL in the routing candidate base. Under OFF they are not —
    which is the LT1.1 r2 receipt (source in mounts: 0/57 correct rows)."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        didx, _ = fold(repo, ids)
        assert not any(a.grafts[i].get('retired') for i in ids)
        base = route_base(a)
        assert set(ids) <= base, (sorted(base), ids)
        assert didx in base          # the digest is ADDED, not substituted
    finally:
        repo.close()


def test_on_lineage_recorded_both_ways(tmp_path, monkeypatch):
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        didx, _ = fold(repo, ids)
        dmeta = a.grafts[didx]['metadata']
        assert dmeta[fr.LINEAGE_RETAINED_SOURCES] == list(ids)
        assert dmeta['fold_lineage'] == fr.LINEAGE_SOURCES_RETAINED
        for i in ids:
            assert a.grafts[i]['metadata'][fr.LINEAGE_DIGEST_OF] == didx
        assert fr.retained_source_indices(a.grafts) == sorted(ids)
    finally:
        repo.close()


def test_on_sources_are_fold_exempt(tmp_path, monkeypatch):
    """Without the exemption the stateless librarian plan re-selects the same
    still-active window forever. `no_fold` is the width guard's own
    anti-reselect channel (SCOUT-FIX-9), reused."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        fold(repo, ids)
        assert all(a.grafts[i]['no_fold'] for i in ids)
        assert not set(repo._foldable(('turn',))) & set(ids)
    finally:
        repo.close()


def test_on_fold_once_records_and_persists_lineage(tmp_path, monkeypatch):
    """Through the REPOSITORY's fold job, not the arena call: the lineage is
    completed into the persisted metadata schema and marked dirty."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        assert repo._fold_once([('digest', list(ids))]) is True
        event = repo.fold_history[-1]
        assert event['accepted'] is True
        assert event['retained_sources'] == list(ids)
        assert event['fold_lineage'] == fr.LINEAGE_SOURCES_RETAINED
        didx = event['digest_idx']
        for i in list(ids) + [didx]:
            meta = a.grafts[i]['metadata']
            # _default_metadata merged UNDER the F1 keys: both present.
            assert meta['active'] is True
            assert 'importance' in meta
            assert i in repo.dirty_nodes
        assert a.grafts[ids[0]]['metadata'][fr.LINEAGE_DIGEST_OF] == didx
        # Payloads are NOT paged out — a retained source is a live reader.
        assert all(a.grafts[i].get('h') is not None for i in ids)
    finally:
        repo.close()


def test_off_fold_once_records_retired_lineage(tmp_path, monkeypatch):
    repo, ids = seeded(tmp_path, monkeypatch, retain=False)
    try:
        assert repo._fold_once([('digest', list(ids))]) is True
        event = repo.fold_history[-1]
        assert event['retained_sources'] == []
        assert event['fold_lineage'] == fr.LINEAGE_SOURCES_RETIRED
        assert all(repo.arena.grafts[i]['retired'] for i in ids)
    finally:
        repo.close()


def test_on_era_fold_also_retains(tmp_path, monkeypatch):
    """An ERA fold (digest of digests) takes the same branch — it is the era
    path whose lost sources produced r2's confabulated Breakwater tuples."""
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        for i in ids:
            a.grafts[i]['kind'] = 'digest'
        didx, _ = fold(repo, ids)
        a.grafts[didx]['kind'] = 'era'
        assert not any(a.grafts[i].get('retired') for i in ids)
        assert set(ids) <= route_base(a)
    finally:
        repo.close()


def test_fidelity_abort_unchanged_under_flag(tmp_path, monkeypatch):
    """A fold that fails MIN_FOLD_KEEP still ABORTS and still leaves the
    sources untouched — retention never rescues a lossy digest."""
    for retain in (False, True):
        repo, ids = seeded(tmp_path / f'a{int(retain)}', monkeypatch,
                           retain=retain)
        try:
            a = repo.arena

            class Lossy(EnumeratedModel):
                def __call__(self, ids_, kv_caches=None, **kw):
                    if kv_caches is None:
                        self.fold_output = (
                            'the archival record notes that several matters '
                            'were discussed and then set aside.<|end|>')
                    return Model.__call__(self, ids_, kv_caches=kv_caches, **kw)

            a.m = Lossy(a.m.codec)
            didx, text = a.consolidate(ids)
            assert didx is None and text is None
            assert a.last_consolidation_result['accepted'] is False
            assert not any(a.grafts[i].get('retired') for i in ids)
            for i in ids:
                assert fr.LINEAGE_DIGEST_OF not in (
                    a.grafts[i].get('metadata') or {})
        finally:
            repo.close()


# --------------------------------------------------- supersession unchanged

@pytest.mark.parametrize('retain', [False, True])
def test_correction_retires_stale_source_and_digest(tmp_path, monkeypatch,
                                                    retain):
    """THE load-bearing assertion. A correction must retire every ROUTABLE
    carrier of the stale value in BOTH states, and the result must never be
    that retention leaves a corrected fact reachable.

    MEASURED, and the asymmetry is the point (probe receipt in the F1
    REPORT): `correct_memory` only considers nodes whose
    `metadata.active` is still True, so
      * OFF supersedes [digest] — the sources were already retired BY THE
        FOLD and are invisible to the correction. Nothing is lost, because
        they were already unroutable.
      * ON  supersedes [stale source, digest] — the source is active again,
        so the correction's own text match reaches it.
    Retention makes a correction reach MORE nodes, never fewer. In both
    states the routing base after the correction contains no node carrying
    the stale value."""
    texts = list(SOURCES)
    repo, ids = seeded(tmp_path / f'c{int(retain)}', monkeypatch,
                       retain=retain, texts=texts)
    try:
        a = repo.arena
        didx, digest_text = fold(repo, ids)
        assert 'BX-44' in digest_text        # the digest holds the stale copy
        new = repo.correct_memory(
            'BX-44', 'The Breakwater beacon sits at grid CX-97, corrected.')
        supersedes = a.grafts[new]['metadata']['supersedes']
        base = route_base(a)
        # The stale value is unroutable in BOTH states.
        assert ids[0] not in base
        assert a.grafts[ids[0]]['retired'] is True
        assert a.grafts[ids[0]]['metadata']['active'] is False
        assert didx not in base
        assert a.grafts[didx]['retired'] is True
        assert a.grafts[didx]['metadata']['superseded_by'] == [new]
        assert not any('BX-44' in a.grafts[i]['text'] for i in base)
        # The asymmetry, pinned explicitly rather than asserted away.
        assert didx in supersedes
        if retain:
            assert ids[0] in supersedes
            assert a.grafts[ids[0]]['metadata']['superseded_by'] == [new]
        else:
            assert ids[0] not in supersedes   # already retired by the fold
        # And the replacement IS routable.
        assert 'CX-97' in a.grafts[new]['text']
        assert new in base
    finally:
        repo.close()


@pytest.mark.parametrize('retain', [False, True])
def test_correction_supersedes_list_is_explicit(tmp_path, monkeypatch, retain):
    """Whatever the flag, the correction's lineage is an EXPLICIT
    `supersedes` list — nothing is retired by inference, and every node on it
    gets the reciprocal `superseded_by` edge. The list may legitimately be
    EMPTY (see the paraphrase test below); what is pinned here is that every
    entry on it is fully superseded, never half."""
    repo, ids = seeded(tmp_path / f's{int(retain)}', monkeypatch,
                       retain=retain)
    try:
        a = repo.arena
        fold(repo, ids)
        new = repo.correct_memory(
            'BX-44', 'The Breakwater beacon sits at grid CX-97, corrected.')
        supersedes = a.grafts[new]['metadata']['supersedes']
        assert supersedes
        for i in supersedes:
            assert a.grafts[i]['retired'] is True
            assert a.grafts[i]['metadata']['active'] is False
            assert a.grafts[i]['metadata']['superseded_by'] == [new]
            assert i not in route_base(a)
    finally:
        repo.close()


@pytest.mark.parametrize('retain', [False, True])
def test_paraphrasing_digest_escapes_correction_in_both_arms(
        tmp_path, monkeypatch, retain):
    """HONEST RED, pinned as a KNOWN PRE-EXISTING GAP — not caused by F1.

    `correct_memory` finds its targets by matching the query string against
    node TEXT (`_native_active_text_matches`, then the Python `q in
    text.lower()` fallback). A fold digest that PARAPHRASES a fact rather
    than copying its wording therefore does not match, and the stale copy
    inside it is never superseded. Measured on this CPU double with a
    multi-word entity ("Iona Vale") that the extractive digest emits as
    separate tokens:

        OFF : supersedes == []   — the correction retires NOTHING. The
              sources were already retired by the fold and the digest does
              not match, so the stale value stays routable inside it.
        ON  : supersedes == [source] — the retained source is still active,
              so the correction reaches it; the digest still escapes.

    So retention does not FIX this gap, and it does not CAUSE it either: in
    both arms the paraphrasing digest survives. In the ON arm strictly more
    of the stale record is retired than in the OFF arm. A1 already carries
    the entity-scoped remedy for its own merged digests
    (`_alias_extend_correction_targets`); generalising that to ordinary fold
    digests is a SUCCESSOR, deliberately out of F1's scope (F1's mission is
    the fold lifecycle, and widening supersession matching is a change to
    correctness semantics the order does not authorise)."""
    repo, ids = seeded(tmp_path / f'p{int(retain)}', monkeypatch,
                       retain=retain)
    try:
        a = repo.arena
        didx, digest_text = fold(repo, ids)
        # The precondition: the digest holds the fact, but not the wording.
        assert 'Iona' in digest_text and 'Vale' in digest_text
        assert 'Iona Vale' not in digest_text
        new = repo.correct_memory(
            'Iona Vale', 'The Commtower maintenance lead is Pell Sorrow now.')
        supersedes = a.grafts[new]['metadata']['supersedes']
        # The gap, in both arms: the digest escapes.
        assert didx not in supersedes
        assert not a.grafts[didx].get('retired')
        assert didx in route_base(a)
        # The difference retention makes: the source IS caught.
        if retain:
            assert supersedes == [ids[2]]
            assert a.grafts[ids[2]]['retired'] is True
            assert ids[2] not in route_base(a)
        else:
            assert supersedes == []
        # And the replacement is routable in both arms.
        assert new in route_base(a)
    finally:
        repo.close()


@pytest.mark.parametrize('retain', [False, True])
def test_forget_unchanged_under_flag(tmp_path, monkeypatch, retain):
    repo, ids = seeded(tmp_path / f'f{int(retain)}', monkeypatch,
                       retain=retain)
    try:
        a = repo.arena
        fold(repo, ids)
        repo.forget('Hull-1')
        assert a.grafts[ids[3]]['retired'] is True
        assert ids[3] not in route_base(a)
    finally:
        repo.close()


# --------------------------------------------------------- A1 is invariant

@pytest.mark.parametrize('retain', [False, True])
def test_alias_merge_lineage_invariant_under_flag(tmp_path, monkeypatch,
                                                  retain):
    """GRM-A1 pins `retain=False`: the alias EDGE is always superseded,
    whatever the window-fold policy says. Resurrecting a bare edge is exactly
    the RD2 defect A1 exists to fix."""
    monkeypatch.setenv('GRM_ALIAS_FOLD_MERGE', '1')
    repo = repository(tmp_path / f'a{int(retain)}', monkeypatch)
    try:
        a = repo.arena
        a.fold_retain_sources = bool(retain)
        assert repo.alias_fold_merge is True

        class AliasModel(Model):
            def __call__(self, ids_, kv_caches=None, **kw):
                if kv_caches is None:
                    prompt = self.codec.decode(ids_[0])
                    lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
                    self.fold_output = (' '.join(lines) + '<|end|>')
                return super().__call__(ids_, kv_caches=kv_caches, **kw)

        a.m = AliasModel(a.m.codec)
        base_idx = a.deposit(
            'The current F1-AliasBase-0 value is Jasper-711 as recorded.')
        edge_idx = a.deposit(
            'F1-Signal-0 is an alias for F1-AliasBase-0.')
        for i in (base_idx, edge_idx):
            repo._ensure_lifecycle(i, a.grafts[i])
        record = repo._alias_fold_once((edge_idx, base_idx))
        if record.get('digest') is None:
            pytest.skip('alias fold aborted on this CPU double: %s'
                        % record.get('reason'))
        # The edge is retired in BOTH states.
        assert a.grafts[edge_idx]['retired'] is True
        assert a.grafts[edge_idx]['metadata']['active'] is False
        assert edge_idx not in route_base(a)
        # And no F1 lineage key leaked onto the alias pair.
        for i in (edge_idx, base_idx):
            meta = a.grafts[i].get('metadata') or {}
            assert fr.LINEAGE_DIGEST_OF not in meta
    finally:
        repo.close()


# ------------------------------------------------- residency accounting

def test_residency_reports_retained_seats_separately(tmp_path, monkeypatch):
    """`seats()` splits retained-source seats out, so a residency bound can
    never be read as 'retention costs nothing' without the split visible."""
    from scripts.grm_c7_run import seats
    repo, ids = seeded(tmp_path, monkeypatch, retain=True)
    try:
        a = repo.arena
        didx, _ = fold(repo, ids)
        a.cur_mounts = [ids[0], didx]
        a.cur_mount_n = 2
        row = seats(a)
        assert row['retained_source_mounted_ids'] == [ids[0]]
        assert row['retained_source_token_seats'] == int(
            a.grafts[ids[0]]['ntok'])
        assert row['retained_source_token_seats'] <= row['summed_token_seats']
    finally:
        repo.close()


def test_residency_off_path_reports_zero(tmp_path, monkeypatch):
    from scripts.grm_c7_run import seats
    repo, ids = seeded(tmp_path, monkeypatch, retain=False)
    try:
        a = repo.arena
        didx, _ = fold(repo, ids)
        a.cur_mounts = [didx]
        a.cur_mount_n = 1
        row = seats(a)
        assert row['retained_source_mounted_ids'] == []
        assert row['retained_source_token_seats'] == 0
    finally:
        repo.close()


# ------------------------------------------------------------ registration

def test_registration_present_and_shaped():
    reg = json.loads(
        (ROOT / 'artifacts/grm_f1/lt1_1_r3/registration.json').read_text())
    assert reg['schema'] == 'grm.lt1_1_r3.registration.v1'
    assert reg['status'] == 'FIT_ESTIMATE'
    assert reg['gpu_executed'] is False
    assert reg['arms'] == ['A']
    assert len(reg['cells']) == 26
    # The order's cap binds MEASURED work; the reservation ceiling stays at
    # r2's, because a budget below the lease sum trips run_cell's rail
    # mid-campaign (LT1.1 amendment 2's own correction). Both are pinned.
    assert reg['order_cap_gpu_hours'] <= 1.3
    assert reg['within_order_cap'] is True
    assert reg['within_budget'] is True
    assert reg['estimate_seconds_sum'] <= reg['order_cap_gpu_seconds']
    assert reg['budget_conflict']['lead_decision_requested']
    names = {f['name'] for f in reg['required_env_flags']}
    assert fr.ENV_NAME in names
    optional = {f['name'] for f in reg['optional_named_flags']}
    assert 'GRM_ALIAS_FOLD_MERGE' in optional
    assert 'GRM_F2_SUPERSEDE_DIGEST_BY_ENTITY' in optional
    # The known gap is REGISTERED, not discovered after the run.
    assert reg['known_gaps'][0]['caused_by_f1'] is False
    # Repo-relative pins only — the round-1 lesson (absolute `wt/` pins
    # became permanent receipts when the forks were pruned). The one
    # allow-listed absolute path is the HF weights snapshot, carried
    # verbatim from r2: an external model cache, not a prunable fork.
    blob = json.dumps(reg).replace(json.dumps(reg['model']['path'])[1:-1], '')
    assert '/mnt/' not in blob
    assert 'wt/' not in blob
    assert '/home/' not in blob
    for key in ('fixture', 'baseline', 'worker'):
        assert not reg[key]['path'].startswith('/')


def test_governing_amendment_rebinds_every_core_pin():
    """The rebind is what makes r3 runnable at all: LT1.1's preflight is
    sha-bound, and F1/F2/F5 all move core inputs.

    Written against the GOVERNING amendment (the highest-numbered one
    carrying a `core_rebind`), not a hard-coded number: amendment 10 was
    F1's, amendment 11 rebinds the merged tree, and a later treatment will
    add another. Every sha is checked against the tree, so the document
    cannot claim one the tree lacks, and the pin set may only GROW.
    """
    import hashlib
    from scripts import grm_lt1_1 as runner
    number, block = runner.governing_core_rebind()
    assert number >= 10
    for name, entry in block['changed'].items():
        on_tree = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert entry['after_sha256'] == on_tree, name
        assert entry['before_sha256'] != on_tree, name
    for name, entry in block['new_inputs'].items():
        assert entry['sha256'] == hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest(), name
    for name, entry in block.get('unchanged', {}).items():
        assert entry['sha256'] == hashlib.sha256(
            (ROOT / name).read_bytes()).hexdigest(), name
    # F1's own input is pinned by the governing rebind, wherever it sits.
    pins = runner.governing_core_pins()
    assert 'core/grm_fold_retain.py' in pins
    assert pins['core/grm_fold_retain.py'] == hashlib.sha256(
        (ROOT / 'core/grm_fold_retain.py').read_bytes()).hexdigest()
    # Nothing is dropped: the pin set only grows across the chain.
    prior = json.loads(
        (ROOT / 'artifacts/grm_d1/lt1_1/amendment8.json').read_text())
    before = (set(prior['core_rebind']['changed'])
              | set(prior['core_rebind']['new_inputs'])
              | set(prior['core_rebind'].get('unchanged', {})))
    assert before <= set(pins)


def test_lead_commands_dry_run_gated():
    """THE gate the order asks for: every runnable command in the lead's
    command file ends in `--dry-run`, so removing it is a deliberate act by
    the lead after reading the registration and never a copy-paste.

    Backslash continuations are joined first — the check is on the LOGICAL
    command, not the physical line, or a `flock ... \\` wrapper would pass
    trivially while the command it wraps went unchecked."""
    path = ROOT / 'artifacts/grm_f1/lead_commands.txt'
    text = path.read_text()
    joined = re.sub(r'\\\s*\n\s*', ' ', text)
    # `export` and `unset` are ENVIRONMENT lines, not commands: an arm that
    # must run with a flag ABSENT (not merely '0') clears it with `unset`,
    # which is the only way to exercise the "flag unset" resolver branch.
    runnable = [l.strip() for l in joined.splitlines()
                if l.strip() and not l.strip().startswith('#')
                and not l.strip().startswith('export')
                and not l.strip().startswith('unset')]
    assert runnable, joined
    for line in runnable:
        assert line.endswith('--dry-run'), line
    # And every command that could spend GPU is under the flock, with the
    # operator's right of way preserved by --wait (never a lock steal).
    for line in runnable:
        if '--resume' in line or '--run' in line:
            assert line.startswith('flock --wait '), line
    assert '/mnt/' not in text and 'wt/' not in text
