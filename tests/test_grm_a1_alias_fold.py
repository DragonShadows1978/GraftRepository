"""GRM-A1 — alias resolution by fold-merge, flag-gated.  CPU gates only.

WHAT THIS FILE IS EVIDENCE FOR, AND WHAT IT IS NOT.
Every test here runs the REAL production objects — ``GraftRepository``,
``ArenaCache`` (via ``scripts.grm_c7_diagnose.CPUArena``, which replaces the
numerical/payload seams only), the real ``consolidate``/FIX-5 prompts, the
real admission and fit, the real ``step`` — against a DETERMINISTIC CPU model
double.  That makes it PLUMBING and EVIDENCE-SUFFICIENCY evidence: which
nodes exist, what lineage they carry, how many seats a digest costs, and
whether the single mount a probe receives CONTAINS the alias -> base -> value
chain.

It is NOT model-quality evidence.  The CPU codec's token lengths are not
GPT-OSS geometry (the registered production widths, 49 + 61 = 110 > 96, are
carried in the fixture and asserted separately as the PRODUCTION arithmetic
that motivates the merge).  Whether GPT-OSS-20B reads a merged digest
correctly is the registered GPU contrast the lead runs; see
``artifacts/grm_a1/lead_commands.txt``.

THE READER DOUBLE AND ITS ONE HONEST LIMIT.
``scripts.grm_c7_diagnose.Model`` answers with the LAST ``current X value is
Y`` it can see, and never reads the question's entity.  The RD2 alias base is
a COMPOUND record naming both ``C7-AliasBase-0`` (Jasper-711) and
``C7-AliasBase-1`` (Jasper-712), so on a correct merged mount that entity-blind
reader returns ``Jasper-712`` for BOTH probes.  That is a limitation of the
stub, not of the retrieval: the evidence is present and complete.  Rather
than hide it, this file scores the C7/RD2 groups on EVIDENCE SUFFICIENCY (the
mounted text contains alias, base and the expected value) and additionally
runs ``AliasResolvingModel`` — a reader that resolves the alias FROM THE
MOUNTED TEXT ALONE, blind to the fixture, its expected answers and the
repository's metadata — to show the chain is followable.  ``test_rd2_``
``fake_reader_entity_blind_limit`` pins the stub's limitation explicitly so
it can never be mistaken for a pass.

ADMISSION RULE — PIN IT OR MEASURE THE WRONG ONE.
``grm_c2_cells.environment()`` strips every ``GRM_*`` key and re-pins a FIXED
set that does NOT include ``GRM_ADMISSION_RULE``, so any check that does not
re-pin the rule AFTERWARDS silently measures the default
``all_tokens_bind``.  LT1 is registered at ``margin_first``
(``scripts/grm_lt1.py:98``, enforced at ``:197``, exported at
``grm_lt1_worker.py:245``; the contract is recorded verbatim at
``scripts/grm_scout_fix6_replay.py:55``).  An earlier revision of this file
measured LT1 serving WITHOUT that pin, recorded 10/10 abstaining
``identifier_unbound``, and reported it as a pre-existing RED.  That reading
was wrong about WHICH RULE was under test and has been RETRACTED — see
``test_lt1_serves_under_its_registered_margin_first_rule`` and
``test_lt1_default_rule_abstention_is_the_rule_not_the_repository``.  Use
``pin_admission_rule()`` for every LT1 serving assertion.

Prior art: ``tests/test_grm_scout_fix5.py`` (GRM contributors, 2026) supplies
the ``EnumeratedModel`` stimulus contract and the repository fixture shape,
reused here verbatim in spirit — the fold output is derived ONLY from the
enumerated ``[source N]`` spans the production prompt itself emits, never
from expected answers.  ``scripts/grm_c7_diagnose.py`` supplies the CPU
doubles.  The rule-pinning discipline is ``grm_scout_fix6_replay``'s
(GRM contributors, 2026), applied verbatim.  No prior art known to me for
this exact alias gate composition.
"""

import json
from pathlib import Path
import re

import pytest

from core import grm_alias_fold as af
from core import grm_legacy_defaults as legacy_defaults
from scripts.grm_c7_diagnose import Model, repository

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------- GRM-D2 re-pin
#
# GRM-D2 (2026-09-11) flipped TWO defaults this suite depends on: A1 itself
# (OFF -> ON) and the admission rule (all_tokens_bind -> margin_first).
# THIS SUITE'S ASSERTIONS ARE UNCHANGED -- it is the receipt for the pre-D2
# world, including the A1-OFF byte-identity proof and the RED that shows
# LT1's abstention belonged to the OLD default rule.  One autouse pin makes
# "unset" mean pre-D2 again for every test here; ``legacy()`` does the same
# for the resolver tests that pass an explicit ``environ=`` dict instead of
# reading ``os.environ``.
#
# Tests below that PIN ``margin_first`` or ``GRM_ALIAS_FOLD_MERGE=1``
# explicitly are unaffected: an explicitly set flag outranks the umbrella
# by construction (core/grm_legacy_defaults.py, precedence rung 2).

def legacy(**extra):
    """An ``environ`` mapping pinned to the pre-D2 defaults."""
    return {legacy_defaults.ENV_NAME: "1", **extra}


@pytest.fixture(autouse=True)
def _grm_d2_legacy_defaults(monkeypatch):
    monkeypatch.setenv(legacy_defaults.ENV_NAME, "1")
FIXTURE = json.loads(
    (ROOT / 'artifacts/grm_a1/alias_fixture.json').read_text())

SYSTEM = ('<|start|>system<|message|>You are ChatGPT. Reasoning: low. '
          'Valid channel: final.<|end|>')


def harmony(user, assistant='Recorded.'):
    """The exact stored-node shape the RD2 receipts record."""
    return (SYSTEM + '<|start|>user<|message|>' + user + '<|end|>'
            + '<|start|>assistant<|channel|>final<|message|>'
            + assistant + '<|end|>')


class EnumeratedModel(Model):
    """FIX-5 stimulus: restate the enumerated source spans as prose.

    Prior art: ``tests/test_grm_scout_fix5.py::EnumeratedModel`` (GRM, 2026),
    same contract.  Reads ONLY the ``[source N]`` lines the production
    ``_consolidation_prompts`` put in front of it — never mounted K/V, never
    the fixture, never an expected answer.  Emits SENTENCES because the
    production QC forbids list-form digests.
    """

    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None:
            prompt = self.codec.decode(ids[0])
            lines = re.findall(r'^\[source \d+\] (.+)$', prompt, re.M)
            if lines:
                self.fold_output = ('The archived record states that '
                                    + ' '.join(l.rstrip('.') + '.'
                                               for l in lines) + '<|end|>')
            else:
                self.fold_output = None
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)


class AliasResolvingModel(EnumeratedModel):
    """Reader that FOLLOWS an alias relation present in its own context.

    Deliberately minimal and fixture-blind: from the visible text it (1)
    finds the questioned name, (2) if that name is declared an alias of
    another name, rewrites the question to the base, and (3) returns that
    entity's ``current … value``.  It never sees ``expected``, the probe
    record, node metadata or the repository.  It exists to answer one
    question the entity-blind stub cannot: *is the alias chain followable
    from the mounted evidence alone?*
    """

    # The token class deliberately excludes a TRAILING '.': identifiers here
    # end sentences, and a greedy `[\w.:-]+` swallows the full stop, so
    # "C7-AliasBase-0." never matches "C7-AliasBase-0" on the next hop.
    _NAME = r'[A-Za-z0-9][\w:-]*(?:\.[A-Za-z0-9][\w:-]*)*'
    _VALUE = re.compile(
        r'current\s+(' + _NAME + r')\s+value\s+is\s+(' + _NAME + r')', re.I)
    _SETTLED = re.compile(
        r"(" + _NAME + r")'s\s+([\w ]+?)\s+(?:will\s+be|is)\s+([^.]+)\.", re.I)

    def __call__(self, ids, kv_caches=None, **kwargs):
        if kv_caches is None and self.fold_output is None:
            visible = self.injected + '\n' + self.codec.decode(ids[0])
            answer = self._resolve(visible)
            if answer is not None:
                self.remaining = self.codec.encode(answer + '<|end|>')
                token = self.remaining.pop(0)
                self.calls.append({'input': self.codec.decode(ids[0]),
                                   'position_offset': 0, 'live_shift': None,
                                   'initial': True, 'output_token': token,
                                   'injected': self.injected})
                import numpy as np
                from types import SimpleNamespace
                logits = np.zeros((1, 1, len(self.codec.words)),
                                  dtype=np.float32)
                logits[0, 0, token] = 1
                return (SimpleNamespace(numpy=lambda: logits),
                        {'tokens': len(ids[0])})
        return super().__call__(ids, kv_caches=kv_caches, **kwargs)

    def _asked_name(self, visible):
        tail = visible.rsplit('<|start|>user<|message|>', 1)[-1]
        ask = re.search(r'value for (' + self._NAME + r')', tail, re.I)
        if ask:
            return ask.group(1), None
        ask = re.search(r"for (?:the )?(" + self._NAME + r")'s ([\w ]+?)\?",
                        tail, re.I)
        if ask:
            return ask.group(1), ask.group(2).strip()
        return None, None

    def _resolve(self, visible):
        name, attribute = self._asked_name(visible)
        if not name:
            return None
        # Follow an alias declaration that is present in the visible text.
        for _ in range(af.MAX_ALIAS_HOPS):
            edge = re.search(
                r'(?<![\w-])' + re.escape(name)
                + r'\s+is\s+an\s+alias\s+for\s+(' + self._NAME + r')',
                visible, re.I)
            if not edge:
                bare = re.sub(r'^the\s+', '', name, flags=re.I).strip()
                edge = re.search(
                    r"call\s+(" + self._NAME + r")\s+['\u2018\u2019\"]?"
                    r"(?:the\s+)?" + re.escape(bare) + r"['\u2018\u2019\"]?",
                    visible, re.I)
                if edge:
                    name = edge.group(1)
                    continue
                break
            name = edge.group(1)
        if attribute is None:
            for entity, value in self._VALUE.findall(visible):
                if entity.casefold() == name.casefold():
                    return value
            return 'UNKNOWN'
        for entity, attr, value in self._SETTLED.findall(visible):
            if (entity.casefold() == name.casefold()
                    and attr.strip().casefold() == attribute.casefold()):
                return value.strip()
        return 'UNKNOWN'


def build(tmp_path, monkeypatch, flag, name='repo', model=EnumeratedModel):
    """A production repository over the CPU doubles, A1 flag as given."""
    monkeypatch.setenv('GRM_ALIAS_FOLD_MERGE', flag)
    repo = repository(tmp_path / name, monkeypatch)
    repo.arena.m = model(repo.arena.m.codec)
    return repo


def seed(repo, texts, *, kind='turn', no_fold=True):
    """Deposit the recorded node texts exactly as the receipts record them.

    ``no_fold=True`` reproduces the RD2 state: every C7 alias edge and base
    carries ``no_fold`` from a prior fidelity abort.  The gate therefore also
    proves the treatment is reachable on nodes the ordinary window fold has
    given up on.
    """
    ids = []
    for text in texts:
        idx = repo.arena.deposit(text)
        repo.arena.grafts[idx]['kind'] = kind
        repo.arena.grafts[idx]['no_fold'] = bool(no_fold)
        ids.append(idx)
    repo._sync_lifecycle()
    return ids


def mounted_text(repo):
    """Concatenated text of whatever the last ``step`` actually mounted."""
    return '\n'.join(str(repo.arena.grafts[int(i)].get('text', ''))
                     for i in (repo.arena.cur_mounts or ()))


def evidence_sufficient(text, *needles):
    """Does the mounted evidence contain every required name/value?"""
    return af.names_present(text, *needles)


# --------------------------------------------------------------- the flag

def test_flag_default_off_and_fails_closed():
    assert af.alias_fold_enabled(environ=legacy()) is False
    assert af.alias_fold_enabled(
        environ=legacy(GRM_ALIAS_FOLD_MERGE='1')) is True
    assert af.alias_fold_enabled(
        environ=legacy(GRM_ALIAS_FOLD_MERGE='on')) is True
    assert af.alias_fold_enabled(
        environ=legacy(GRM_ALIAS_FOLD_MERGE='0')) is False
    # Unknown token fails CLOSED, matching L2 / A-DEC precedent.  GRM-D2
    # moved the rung it closes ON -- the override helper now returns None
    # and the SHIPPED default decides -- but under this suite's legacy pin
    # the shipped default is OFF, so the verdict is the same.
    assert af.alias_fold_enabled(
        environ=legacy(GRM_ALIAS_FOLD_MERGE='maybe')) is False
    # Explicit caller wins over the environment, in both directions.
    assert af.alias_fold_enabled(
        True, environ=legacy(GRM_ALIAS_FOLD_MERGE='0')) is True
    assert af.alias_fold_enabled(
        False, environ=legacy(GRM_ALIAS_FOLD_MERGE='1')) is False


def test_flag_off_byte_identical(tmp_path, monkeypatch):
    """OFF must be byte-identical to the pre-A1 branch.

    The comparison is against the flag being ABSENT from the environment
    entirely (the pre-A1 world, where no such variable existed) versus
    explicitly ``0``.  Every observable the A1 code could touch is captured:
    the full node table, every metadata field, the fold history, the
    consolidation receipts, the served answers and the model's own call log
    (inputs, outputs, injected text, position offsets) — so a changed PROMPT
    or a changed number of generation steps would fail this too.
    """
    records = {}
    for label, flag in (('absent', None), ('explicit_off', '0')):
        monkeypatch.delenv('GRM_ALIAS_FOLD_MERGE', raising=False)
        if flag is not None:
            monkeypatch.setenv('GRM_ALIAS_FOLD_MERGE', flag)
        repo = repository(tmp_path / label, monkeypatch)
        repo.arena.m = EnumeratedModel(repo.arena.m.codec)
        try:
            assert repo.alias_fold_merge is False
            rows = FIXTURE['rd2']['rows']
            seed(repo, [FIXTURE['rd2']['base_text']]
                 + [r['edge_text'] for r in rows[:2]])
            repo.arena.m.fold_output = None
            answers = [repo.arena.step(r['question'], ngen=8, deposit=False)[0]
                       for r in rows[:2]]
            records[label] = {
                'answers': answers,
                'nodes': [{k: v for k, v in g.items()
                           if k in ('text', 'kind', 'retired', 'sources',
                                    'ntok', 'no_fold', 'metadata',
                                    'child_cents')
                           and k != 'child_cents'}
                          for g in repo.arena.grafts],
                'fold_history': repo.fold_history,
                'alias_history': repo.alias_fold_history,
                'calls': repo.arena.m.calls,
                'consolidation': getattr(
                    repo.arena, 'last_consolidation_result', None),
            }
        finally:
            repo.close()
    left = json.dumps(records['absent'], indent=1, sort_keys=True, default=str)
    right = json.dumps(records['explicit_off'], indent=1, sort_keys=True,
                       default=str)
    assert left == right
    # And the A1 code really is inert: no alias decision was even recorded.
    assert records['absent']['alias_history'] == []


def test_flag_off_never_merges_a_real_alias(tmp_path, monkeypatch):
    """The OFF arm reproduces the RD2 defect rather than silently fixing it."""
    repo = build(tmp_path, monkeypatch, '0')
    try:
        rows = FIXTURE['rd2']['rows']
        seed(repo, [FIXTURE['rd2']['base_text'], rows[0]['edge_text']])
        assert repo._alias_fold_jobs() == ()
        assert repo.alias_fold_pending() == 0
        assert repo.alias_fold_pass() == []
        before = len(repo.arena.grafts)
        repo.runtime._guard_deposit_width(repo._snapshot_state())
        assert len(repo.arena.grafts) == before
        assert all(not g.get('retired') for g in repo.arena.grafts)
        assert repo.alias_fold_history == []
    finally:
        repo.close()


# ------------------------------------------------------- production widths

def test_production_width_arithmetic_is_the_motivation():
    """The registered RD2 widths: 49 + 61 = 110 > 96. Co-mount is impossible."""
    w = FIXTURE['production_widths']
    assert w['alias_edge_ntok'] == 49 and w['alias_base_ntok'] == 61
    assert w['edge_plus_base'] == w['alias_edge_ntok'] + w['alias_base_ntok']
    assert w['edge_plus_base'] == 110 > w['arena_width'] == 96
    assert w['over_width_by'] == w['edge_plus_base'] - w['arena_width'] == 14
    for row in FIXTURE['rd2']['rows']:
        assert row['edge_ntok_production'] == 49
        assert row['recorded_mounted_ids'] == [row['edge_id']]
        assert 'sorry' in row['recorded_served']


def test_width_receipt_is_taken_before_any_single_mount_claim():
    encode = lambda t: t.split()
    fits = af.width_receipt(encode, 'a b c', 96)
    assert fits == {'digest_tokens': 3, 'arena_width': 96,
                    'fits': True, 'single_mount_claimable': True}
    over = af.width_receipt(encode, ' '.join(['x'] * 200), 96)
    assert over['digest_tokens'] == 200
    assert over['fits'] is False and over['single_mount_claimable'] is False


# ------------------------------------------------------------- RD2 group 8

def _rd2_repo(tmp_path, monkeypatch, flag, model=EnumeratedModel, name='rd2'):
    repo = build(tmp_path, monkeypatch, flag, name=name, model=model)
    texts = [FIXTURE['rd2']['base_text']] + [
        r['edge_text'] for r in FIXTURE['rd2']['rows'][:2]]
    seed(repo, texts)
    return repo


@pytest.mark.parametrize('row', FIXTURE['rd2']['rows'],
                         ids=[r['probe_id'] for r in FIXTURE['rd2']['rows']])
def test_rd2_alias_red_before_flag_off(tmp_path, monkeypatch, row):
    """RED: with A1 OFF the probe mounts the bare edge and cannot answer."""
    repo = _rd2_repo(tmp_path, monkeypatch, '0', name=row['probe_id'] + '-off')
    try:
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        text = mounted_text(repo)
        assert 'is an alias for' in text
        assert not evidence_sufficient(text, row['expected']), (
            'OFF arm must NOT have the value in the mount', text)
    finally:
        repo.close()


@pytest.mark.parametrize('row', FIXTURE['rd2']['rows'],
                         ids=[r['probe_id'] for r in FIXTURE['rd2']['rows']])
def test_rd2_alias_green_after_flag_on(tmp_path, monkeypatch, row):
    """GREEN: one merged digest carries alias, base and value, under width."""
    repo = _rd2_repo(tmp_path, monkeypatch, '1', name=row['probe_id'] + '-on')
    try:
        merges = repo.alias_fold_pass()
        assert merges, 'the flag-ON pass must produce a merge'
        assert all(m['reason'] == af.REASON_MERGED for m in merges), merges
        alias = row['question'].split('for ')[1].split('?')[0]
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        mounts = list(repo.arena.cur_mounts or ())
        assert len(mounts) == 1, ('single-mount claim', mounts)
        text = mounted_text(repo)
        # The mounted evidence carries the whole chain: alias, base, value.
        base_entity = 'C7-AliasBase-' + alias.rsplit('-', 1)[1]
        assert evidence_sufficient(text, alias, base_entity, row['expected'])
        # ... and it fits the arena width the CPU double declares.
        seats = int(repo.arena.grafts[mounts[0]]['ntok'])
        assert seats <= int(repo.arena.width)
    finally:
        repo.close()


@pytest.mark.parametrize('row', FIXTURE['rd2']['rows'],
                         ids=[r['probe_id'] for r in FIXTURE['rd2']['rows']])
def test_rd2_alias_chain_is_followable_from_the_mount(tmp_path, monkeypatch,
                                                      row):
    """A fixture-blind reader resolves the alias from the mounted text alone."""
    repo = _rd2_repo(tmp_path, monkeypatch, '1', model=AliasResolvingModel,
                     name=row['probe_id'] + '-chain')
    try:
        assert repo.alias_fold_pass()
        repo.arena.m.fold_output = None
        answer, _ = repo.arena.step(row['question'], ngen=8, deposit=False)
        assert answer.strip() == row['expected'], (answer, mounted_text(repo))
    finally:
        repo.close()


def test_rd2_fake_reader_entity_blind_limit(tmp_path, monkeypatch):
    """HONEST LIMIT, pinned: the c7_diagnose stub cannot pick among two values.

    On a CORRECT merged mount the entity-blind ``Model`` returns the LAST
    ``current X value is Y`` it sees — ``Jasper-712`` for both Signal-0 and
    Signal-1, because the RD2 base is one compound record.  This is a
    property of the CPU double, not of the retrieval, and it is asserted here
    so no reader of this file can mistake the evidence-sufficiency gates
    above for a reading-quality claim.
    """
    repo = _rd2_repo(tmp_path, monkeypatch, '1', name='blindlimit')
    try:
        assert repo.alias_fold_pass()
        repo.arena.m.fold_output = None
        answers = []
        for row in FIXTURE['rd2']['rows'][:2]:
            answers.append(
                repo.arena.step(row['question'], ngen=8, deposit=False)[0].strip())
        assert answers == ['Jasper-712', 'Jasper-712']
        # Both values ARE in the mount; the stub simply cannot choose.
        text = mounted_text(repo)
        assert evidence_sufficient(text, 'Jasper-711')
        assert evidence_sufficient(text, 'Jasper-712')
    finally:
        repo.close()


# ------------------------------------------------------------ C7 r3 group

C7_ANSWERABLE = [r for r in FIXTURE['c7_r3']['rows'] if r['answerable']]
C7_CONTROLS = [r for r in FIXTURE['c7_r3']['rows'] if not r['answerable']]


def _c7_repo(tmp_path, monkeypatch, flag, name, model=EnumeratedModel):
    """Seed the C7 r3 alias world from the fixture's own oracle source texts."""
    repo = build(tmp_path, monkeypatch, flag, name=name, model=model)
    texts, seen = [], set()
    for row in FIXTURE['c7_r3']['rows']:
        for src in row['oracle_source_texts']:
            if src not in seen:
                seen.add(src)
                texts.append(harmony(src))
    seed(repo, texts)
    return repo


@pytest.mark.parametrize('row', C7_ANSWERABLE,
                         ids=[r['probe_id'] for r in C7_ANSWERABLE])
def test_c7_alias_red_before_flag_off(tmp_path, monkeypatch, row):
    repo = _c7_repo(tmp_path, monkeypatch, '0', row['probe_id'] + '-off')
    try:
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        assert not evidence_sufficient(mounted_text(repo), row['expected'])
    finally:
        repo.close()


@pytest.mark.parametrize('row', C7_ANSWERABLE,
                         ids=[r['probe_id'] for r in C7_ANSWERABLE])
def test_c7_alias_green_after_flag_on(tmp_path, monkeypatch, row):
    repo = _c7_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-on')
    try:
        merges = repo.alias_fold_pass()
        assert merges
        alias = row['question'].split('for ')[1].split('?')[0]
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        text = mounted_text(repo)
        assert evidence_sufficient(text, alias, row['expected'])
        for idx in (repo.arena.cur_mounts or ()):
            assert int(repo.arena.grafts[int(idx)]['ntok']) <= int(
                repo.arena.width)
    finally:
        repo.close()


@pytest.mark.parametrize('row', C7_CONTROLS,
                         ids=[r['probe_id'] for r in C7_CONTROLS])
def test_c7_unanswerable_alias_control_unchanged(tmp_path, monkeypatch, row):
    """CONTROL: C7-Signal-2's base has no value; the merge must not invent one.

    ``C7-AliasBase-2`` is never given a value in the fixture, so the edge for
    Signal-2 must resolve to ``alias_base_missing`` and leave the edge
    exactly where it was.  The probe's expected answer stays UNKNOWN.
    """
    repo = _c7_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-ctl')
    try:
        repo.alias_fold_pass()
        assert row['expected'] == 'UNKNOWN'
        reasons = {d['reason'] for d in repo.alias_fold_history}
        assert af.REASON_BASE_MISSING in reasons, repo.alias_fold_history
        # No digest anywhere claims to have merged Signal-2.
        for _, merged in repo._alias_merged_digests():
            assert 'signal-2' not in str(merged.get('alias', '')).casefold()
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        assert not evidence_sufficient(mounted_text(repo), 'AliasBase-2 value')
    finally:
        repo.close()


# ---------------------------------------------------------------- LT1 group

LT1_ROWS = FIXTURE['lt1']['rows']


def _lt1_repo(tmp_path, monkeypatch, flag, name, model=EnumeratedModel):
    repo = build(tmp_path, monkeypatch, flag, name=name, model=model)
    texts, seen = [], set()
    for row in LT1_ROWS:
        for src in row['source_texts']:
            if src not in seen:
                seen.add(src)
                texts.append(harmony(src))
    seed(repo, texts)
    return repo


@pytest.mark.parametrize('row', LT1_ROWS, ids=[r['probe_id'] for r in LT1_ROWS])
def test_lt1_alias_red_before_flag_off(tmp_path, monkeypatch, row):
    """RED: with A1 OFF no node joins the alias to its value, and no mount does.

    Measured under LT1's REGISTERED ``margin_first`` rule, so the OFF arm is
    compared to the ON arm on the same footing — an unpinned measurement here
    would abstain for a reason that has nothing to do with the flag and would
    pass for the wrong reason.
    """
    repo = _lt1_repo(tmp_path, monkeypatch, '0', row['probe_id'] + '-off')
    try:
        assert repo.alias_fold_pass() == []
        alias = re.search(r"for the (\w+)'s", row['question']).group(1)
        for graft in repo.arena.grafts:
            assert not af.names_present(str(graft.get('text', '')),
                                        alias, row['expected'])
        pin_admission_rule(monkeypatch)
        repo.arena.m.fold_output = None
        repo.arena.step(row['question'], ngen=8, deposit=False)
        # Admitted and mounted under margin_first, but the mount is a bare
        # turn node that cannot join the alias to the value.
        assert not evidence_sufficient(mounted_text(repo), alias,
                                       row['expected'])
    finally:
        repo.close()


@pytest.mark.parametrize('row', LT1_ROWS, ids=[r['probe_id'] for r in LT1_ROWS])
def test_lt1_alias_green_after_flag_on(tmp_path, monkeypatch, row):
    """LT1's aliases are NATURAL language ("Let's call Kestrel 'the Hauler'").

    GREEN here is the merge and its receipt: one digest, naming alias, base
    and value, under the arena width.  The SERVED side is covered by
    ``test_lt1_serves_under_its_registered_margin_first_rule`` and
    ``test_lt1_margin_first_serves_exact_from_the_single_mount`` below, which
    pin LT1's registered ``margin_first`` rule and get 10/10 admitted,
    single-mount, exact.
    """
    repo = _lt1_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-on')
    try:
        merges = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED]
        assert merges, repo.alias_fold_history
        alias = re.search(r"for the (\w+)'s", row['question']).group(1)
        hit = [m for m in merges
               if af.identifier_set(alias) <= af.identifier_set(m['alias'])]
        assert hit, (alias, merges)
        digest = repo.arena.grafts[hit[0]['digest']]
        # One digest carries alias, base and value ...
        assert af.names_present(digest['text'], alias, hit[0]['base_name'],
                                row['expected'])
        # ... and it is single-mount claimable under the arena width.
        assert hit[0]['width']['fits'] is True
        assert hit[0]['width']['single_mount_claimable'] is True
        assert int(digest['ntok']) <= int(repo.arena.width)
        assert hit[0]['width']['digest_tokens'] == int(digest['ntok'])
    finally:
        repo.close()


#: LT1's REGISTERED admission rule.  ``scripts/grm_lt1.py:98`` sets
#: ``effective_admission_rule = 'margin_first'`` and ``:197`` STOPS the run
#: unless the environment agrees; ``scripts/grm_lt1_worker.py:245`` exports it.
#: ``scripts/grm_scout_fix6_replay.py:55`` records the pinning contract in
#: one line: "Pin GRM_ADMISSION_RULE after environment(flags), which removes
#: ambient GRM variables."  ``grm_c2_cells.environment()`` strips every
#: ``GRM_*`` key and re-pins a FIXED set that does NOT include
#: ``GRM_ADMISSION_RULE`` — so a check that does not re-pin it afterwards
#: silently measures the DEFAULT ``all_tokens_bind`` rule instead of LT1's.
LT1_ADMISSION_RULE = 'margin_first'


def pin_admission_rule(monkeypatch, rule=LT1_ADMISSION_RULE):
    """Pin GRM_ADMISSION_RULE the way the R1/C2 workers do: LAST.

    ``monkeypatch`` restores the previous environment at teardown, which is
    the "restore the env after" half of the contract.
    """
    monkeypatch.setenv('GRM_ADMISSION_RULE', rule)


@pytest.mark.parametrize('row', LT1_ROWS, ids=[r['probe_id'] for r in LT1_ROWS])
def test_lt1_serves_under_its_registered_margin_first_rule(
        tmp_path, monkeypatch, row):
    """CORRECTED READING (lead, 2026-09-10).  Supersedes an earlier RED.

    An earlier version of this file measured LT1 serving WITHOUT pinning
    ``GRM_ADMISSION_RULE``, so it ran under today's default
    ``all_tokens_bind`` and recorded 10/10 abstaining ``identifier_unbound``.
    That was reported as "RED, pre-existing, not A1's".  The reading was
    WRONG ABOUT WHICH RULE WAS UNDER TEST: LT1 is registered at
    ``margin_first`` (FIX-6), where the identifier is only a TIE-BREAKER and
    ``margin_first_plan`` never takes a zero-hit abstention branch at all.

    Re-measured with the rule pinned, on BOTH arms (an ideal hand-written
    merged digest, and A1's own merged digest), all ten probes are ADMITTED
    and each mounts exactly one merged digest:

        branch = fit_margin_decisive_rank1,  mounts = [<one digest>]

    So the abstention was an artifact of the unpinned measurement, not a
    property of the repository.  RED item 1 is RETRACTED.
    """
    repo = _lt1_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-mf')
    try:
        merged = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED]
        assert merged
        pin_admission_rule(monkeypatch)
        repo.arena.m.fold_output = None
        _, info = repo.arena.step(row['question'], ngen=12, deposit=False)
        assert info.get('abstained') is not True, info
        assert info.get('abstain_reason') is None, info
        mounts = list(repo.arena.cur_mounts or ())
        assert len(mounts) == 1, (mounts, info)
        assert repo.arena.grafts[mounts[0]].get('kind') == 'digest'
        assert mounts[0] in {m['digest'] for m in merged}
        # The single mount carries the whole chain.
        alias = re.search(r"for the (\w+)'s", row['question']).group(1)
        assert evidence_sufficient(mounted_text(repo), alias, row['expected'])
    finally:
        repo.close()


@pytest.mark.parametrize('row', LT1_ROWS, ids=[r['probe_id'] for r in LT1_ROWS])
def test_lt1_margin_first_serves_exact_from_the_single_mount(
        tmp_path, monkeypatch, row):
    """Under margin_first, the fixture-blind reader answers from the mount.

    The ``grm_c7_diagnose`` stub returns UNKNOWN here — its regex is
    ``current X value is Y`` and LT1's prose is ``X's attr will be Y``.  That
    is the stub's shape, not the repository's: it returns UNKNOWN on the
    IDEAL hand-written digest too (see
    ``test_lt1_margin_first_ideal_and_merged_arms_agree``).
    """
    repo = _lt1_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-mfx',
                     model=AliasResolvingModel)
    try:
        assert [m for m in repo.alias_fold_pass()
                if m['reason'] == af.REASON_MERGED]
        pin_admission_rule(monkeypatch)
        repo.arena.m.fold_output = None
        answer, info = repo.arena.step(row['question'], ngen=12, deposit=False)
        assert len(list(repo.arena.cur_mounts or ())) == 1, info
        assert answer.strip() == row['expected'], (answer, mounted_text(repo))
    finally:
        repo.close()


def test_lt1_margin_first_ideal_and_merged_arms_agree(tmp_path, monkeypatch):
    """The A1 merge matches an IDEAL hand-written digest, probe for probe.

    The ideal arm is built with the flag OFF and hand-written digests, so it
    measures the CEILING the merge is trying to reach.  Agreement on both
    admission and served value is what licenses the claim that A1's digest is
    as good as the best digest anyone could write for these rows.
    """
    ideal_texts = [
        harmony("Kestrel's cargo allowance will be 37 crates."),
        harmony("Lantern's launch date will be 18 October 2196."),
        'ARCHIVE NOTE. the Hauler is an alias for Kestrel. '
        "Kestrel's cargo allowance will be 37 crates.\n",
        'ARCHIVE NOTE. the Beacon is an alias for Lantern. '
        "Lantern's launch date will be 18 October 2196.\n",
    ]

    def arm(name, flag, seeds, merge):
        repo = build(tmp_path, monkeypatch, flag, name=name,
                     model=AliasResolvingModel)
        try:
            seed(repo, seeds)
            if merge:
                assert [m for m in repo.alias_fold_pass()
                        if m['reason'] == af.REASON_MERGED]
            pin_admission_rule(monkeypatch)
            repo.arena.m.fold_output = None
            out = []
            for row in LT1_ROWS:
                answer, info = repo.arena.step(row['question'], ngen=12,
                                               deposit=False)
                out.append((row['probe_id'],
                            info.get('abstained') is not True,
                            len(list(repo.arena.cur_mounts or ())),
                            answer.strip()))
            return out
        finally:
            repo.close()

    lt1_seeds, seen = [], set()
    for row in LT1_ROWS:
        for src in row['source_texts']:
            if src not in seen:
                seen.add(src)
                lt1_seeds.append(harmony(src))

    ideal = arm('mf-ideal', '0', ideal_texts, merge=False)
    mergd = arm('mf-merged', '1', lt1_seeds, merge=True)
    assert ideal == mergd, (ideal, mergd)
    assert all(admitted and mounts == 1 for _, admitted, mounts, _ in mergd)
    assert [a for _, _, _, a in mergd] == [r['expected'] for r in LT1_ROWS]


def test_lt1_default_rule_abstention_is_the_rule_not_the_repository(
        tmp_path, monkeypatch):
    """Why the earlier RED was wrong: the abstention is the RULE's, not A1's.

    Under ``all_tokens_bind`` the same repository abstains, and it abstains
    IDENTICALLY with an ideal hand-written digest.  Pinning ``margin_first``
    — LT1's registered rule — admits both.  Holding the repository fixed
    and moving only the rule is what isolates the cause.

    GRM-D2 (2026-09-11) made ``margin_first`` the shipped default precisely
    because of this RED and LT1's 0/35.  ``rule is None`` below therefore
    means "the PRE-D2 default", which the module's ``GRM_LEGACY_DEFAULTS=1``
    pin supplies; the assertions are unchanged, and the comparison they
    make — same repository, one rule apart — is the whole point.
    """
    lt1_seeds, seen = [], set()
    for row in LT1_ROWS:
        for src in row['source_texts']:
            if src not in seen:
                seen.add(src)
                lt1_seeds.append(harmony(src))
    question = LT1_ROWS[0]['question']

    def observe(rule):
        repo = build(tmp_path, monkeypatch, '1', name='rule-' + str(rule))
        try:
            seed(repo, lt1_seeds)
            assert [m for m in repo.alias_fold_pass()
                    if m['reason'] == af.REASON_MERGED]
            if rule is None:
                monkeypatch.delenv('GRM_ADMISSION_RULE', raising=False)
            else:
                pin_admission_rule(monkeypatch, rule)
            repo.arena.m.fold_output = None
            _, info = repo.arena.step(question, ngen=12, deposit=False)
            return (info.get('abstained') is True,
                    info.get('abstain_reason'),
                    info.get('admission_policy_branch'),
                    len(list(repo.arena.cur_mounts or ())))
        finally:
            repo.close()

    assert observe(None) == (True, 'identifier_unbound',
                             'ambiguous_zero_identifier_hits_k3', 0)
    abstained, reason, branch, mounts = observe('margin_first')
    assert abstained is False and reason is None
    assert branch == 'fit_margin_decisive_rank1' and mounts == 1
    # And the production resolver agrees the pinned value is what is in force.
    monkeypatch.setenv('GRM_ADMISSION_RULE', 'margin_first')
    from core.grm_admission import admission_rule
    assert admission_rule() == 'margin_first'


@pytest.mark.parametrize('row', LT1_ROWS, ids=[r['probe_id'] for r in LT1_ROWS])
def test_lt1_alias_chain_is_followable(tmp_path, monkeypatch, row):
    """The chain is followable from the merged digest's TEXT.

    Because the probe abstains before mounting (see the RED test above), the
    reader is given the merged digest's text directly.  That measures exactly
    what the merge is responsible for — is the evidence complete and
    self-contained? — and claims nothing about admission, which is the part
    that is RED.
    """
    repo = _lt1_repo(tmp_path, monkeypatch, '1', row['probe_id'] + '-chain',
                     model=AliasResolvingModel)
    try:
        merges = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED]
        alias = re.search(r"for the (\w+)'s", row['question']).group(1)
        hit = [m for m in merges
               if af.identifier_set(alias) <= af.identifier_set(m['alias'])]
        assert hit, (alias, merges)
        digest_text = repo.arena.grafts[hit[0]['digest']]['text']
        visible = (digest_text + '\n<|start|>user<|message|>'
                   + row['question'] + '<|end|>')
        assert repo.arena.m._resolve(visible) == row['expected'], visible
    finally:
        repo.close()


# ------------------------------------------------------- revision semantics

def test_base_correction_supersedes_the_merged_digest(tmp_path, monkeypatch):
    """A later correction of the base value must retire the stale merge."""
    repo = build(tmp_path, monkeypatch, '1', name='revision')
    try:
        seed(repo, [harmony('The current C7-AliasBase-0 value is Jasper-711.'),
                    harmony('C7-Signal-0 is an alias for C7-AliasBase-0.')])
        merges = repo.alias_fold_pass()
        assert merges and merges[0]['reason'] == af.REASON_MERGED
        digest = merges[0]['digest']
        assert 'jasper-711' in af.identifier_set(
            repo.arena.grafts[digest]['text'])
        repo.correct_memory(
            'current C7-AliasBase-0 value is Jasper-711',
            'The current C7-AliasBase-0 value is Onyx-911.')
        # The stale merged digest is retired and points at its replacement.
        assert repo.arena.grafts[digest]['retired'] is True
        assert repo.arena.grafts[digest]['metadata']['active'] is False
        assert repo.arena.grafts[digest]['metadata']['superseded_by']
        # The stale value is no longer servable from any ACTIVE node.
        active = ' '.join(str(g.get('text', '')) for g in repo.arena.grafts
                          if not g.get('retired')
                          and (g.get('metadata') or {}).get('active', True))
        assert 'jasper-711' not in af.identifier_set(active)
        assert 'onyx-911' in af.identifier_set(active)
    finally:
        repo.close()


def test_missing_base_leaves_the_edge_alone(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='nobase')
    try:
        ids = seed(repo, [harmony('C7-Signal-9 is an alias for C7-Nowhere-9.')])
        assert repo.alias_fold_pass() == []
        record = repo.alias_fold_history[-1]
        assert record['reason'] == af.REASON_BASE_MISSING
        assert record['edge'] == ids[0] and record['base'] is None
        g = repo.arena.grafts[ids[0]]
        assert not g.get('retired') and g['metadata']['active'] is True
        assert len(repo.arena.grafts) == 1
    finally:
        repo.close()


def test_alias_of_alias_resolves_to_the_terminal_base(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='chain')
    try:
        seed(repo, [harmony('The current C7-Deep-0 value is Jasper-777.'),
                    harmony('C7-Mid-0 is an alias for C7-Deep-0.'),
                    harmony('C7-Top-0 is an alias for C7-Mid-0.')])
        merges = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED]
        assert merges
        tops = [m for m in merges if m['alias'] == 'C7-Top-0']
        assert tops, merges
        # The TERMINAL base's value reached the top alias's digest.
        text = repo.arena.grafts[tops[0]['digest']]['text']
        assert af.names_present(text, 'C7-Top-0', 'Jasper-777')
    finally:
        repo.close()


def test_alias_cycle_refuses_with_a_receipt(tmp_path, monkeypatch):
    """A -> B -> A must REFUSE, never pick an arbitrary member of the cycle."""
    repo = build(tmp_path, monkeypatch, '1', name='cycle')
    try:
        seed(repo, [harmony('C7-Loop-A is an alias for C7-Loop-B.'),
                    harmony('C7-Loop-B is an alias for C7-Loop-A.')])
        assert repo.alias_fold_pass() == []
        reasons = {d['reason'] for d in repo.alias_fold_history}
        assert reasons == {af.REASON_CYCLE}, repo.alias_fold_history
        assert all(not g.get('retired') for g in repo.arena.grafts)
        assert len(repo.arena.grafts) == 2
    finally:
        repo.close()


def test_self_alias_refuses(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='self')
    try:
        seed(repo, [harmony('C7-Same-0 is an alias for C7-Same-0.')])
        assert repo.alias_fold_pass() == []
        assert repo.alias_fold_history[-1]['reason'] == af.REASON_SELF
        assert len(repo.arena.grafts) == 1
    finally:
        repo.close()


def test_alias_reassignment_retires_the_old_digest(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='reassign')
    try:
        seed(repo, [harmony('The current C7-Old-0 value is Jasper-100.'),
                    harmony('C7-Sig-0 is an alias for C7-Old-0.')])
        first = [m for m in repo.alias_fold_pass()
                 if m['reason'] == af.REASON_MERGED]
        assert first
        old_digest = first[0]['digest']
        seed(repo, [harmony('The current C7-New-0 value is Onyx-200.'),
                    harmony('C7-Sig-0 is an alias for C7-New-0.')])
        second = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED]
        assert second
        assert old_digest in second[-1]['reassignment_retired'], second
        assert repo.arena.grafts[old_digest]['retired'] is True
        assert repo.arena.grafts[old_digest]['metadata'][
            'alias_reassigned'] is True
        new_digest = second[-1]['digest']
        assert af.names_present(repo.arena.grafts[new_digest]['text'],
                                'C7-Sig-0', 'Onyx-200')
    finally:
        repo.close()


def test_merge_is_idempotent(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='idem')
    try:
        seed(repo, [harmony('The current C7-Idem-0 value is Jasper-300.'),
                    harmony('C7-IdemSig-0 is an alias for C7-Idem-0.')])
        assert len([m for m in repo.alias_fold_pass()
                    if m['reason'] == af.REASON_MERGED]) == 1
        count = len(repo.arena.grafts)
        for _ in range(3):
            assert repo.alias_fold_pass() == []
        assert len(repo.arena.grafts) == count
    finally:
        repo.close()


# ------------------------------------------------------------ lineage rules

def test_full_coverage_retires_the_base(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='lineage-full')
    try:
        base, edge = seed(
            repo, [harmony('The current C7-Cov-0 value is Jasper-400.'),
                   harmony('C7-CovSig-0 is an alias for C7-Cov-0.')])
        record = [m for m in repo.alias_fold_pass()
                  if m['reason'] == af.REASON_MERGED][0]
        assert record['lineage'] == af.LINEAGE_EDGE_AND_BASE
        assert record['base_identifiers_missing'] == []
        assert repo.arena.grafts[edge]['retired'] is True
        assert repo.arena.grafts[base]['retired'] is True
        digest = repo.arena.grafts[record['digest']]
        assert sorted(digest['metadata']['supersedes']) == sorted([edge, base])
    finally:
        repo.close()


def test_partial_coverage_keeps_the_base_active(tmp_path, monkeypatch):
    """The base survives when the digest lost even ONE of its identifiers.

    Stricter than the fold's own ``MIN_FOLD_KEEP`` on purpose: retiring the
    only active record of a fact the digest dropped is the unrecoverable
    failure the 2026-06-11 fidelity gate exists to prevent.
    """
    class LossyModel(EnumeratedModel):
        def __call__(self, ids, kv_caches=None, **kwargs):
            out = super().__call__(ids, kv_caches=kv_caches, **kwargs)
            if kv_caches is None and self.fold_output:
                # Drop ONE base identifier while keeping alias, base name and
                # the primary value, so the fold's own coverage bar passes.
                self.fold_output = self.fold_output.replace(
                    ' The current C7-Part-0 spare value is Spare-999.', '')
                self.remaining = self.codec.encode(self.fold_output)
            return out

    repo = build(tmp_path, monkeypatch, '1', name='lineage-part',
                 model=LossyModel)
    try:
        base, edge = seed(
            repo,
            [harmony('The current C7-Part-0 value is Jasper-500. '
                     'The current C7-Part-0 spare value is Spare-999.'),
             harmony('C7-PartSig-0 is an alias for C7-Part-0.')])
        records = [m for m in repo.alias_fold_pass()
                   if m['reason'] in (af.REASON_MERGED,
                                      af.REASON_DIGEST_OVER_WIDTH)]
        assert records, repo.alias_fold_history
        record = records[0]
        assert record['lineage'] == af.LINEAGE_EDGE_ONLY
        assert 'spare-999' in record['base_identifiers_missing']
        assert repo.arena.grafts[edge]['retired'] is True
        # The base stays ACTIVE: it is the only record of the dropped fact.
        assert repo.arena.grafts[base]['retired'] is False
        assert repo.arena.grafts[base]['metadata']['active'] is True
        assert repo.arena.grafts[record['digest']][
            'metadata']['supersedes'] == [edge]
    finally:
        repo.close()


def test_coverage_helper_is_exact_not_fractional():
    proved, missing = af.coverage_proves_base_survives(
        'X-1 and X-2 are here', 'X-1 X-2')
    assert proved is True and missing == []
    proved, missing = af.coverage_proves_base_survives('only X-1', 'X-1 X-2')
    assert proved is False and missing == ['x-2']


# -------------------------------------------------------- width guard path

def test_over_width_digest_goes_through_the_existing_split_path(
        tmp_path, monkeypatch):
    """An over-width digest is NOT a failure: it takes the LSR-P2C split."""
    class VerboseModel(EnumeratedModel):
        def __call__(self, ids, kv_caches=None, **kwargs):
            out = super().__call__(ids, kv_caches=kv_caches, **kwargs)
            if kv_caches is None and self.fold_output:
                filler = ' '.join(
                    f'Context note {n} records routine detail.'
                    for n in range(140))
                self.fold_output = self.fold_output.replace(
                    '<|end|>', ' ' + filler + '<|end|>')
                self.remaining = self.codec.encode(self.fold_output)
            return out

    repo = build(tmp_path, monkeypatch, '1', name='overwidth',
                 model=VerboseModel)
    try:
        seed(repo, [harmony('The current C7-Wide-0 value is Jasper-600.'),
                    harmony('C7-WideSig-0 is an alias for C7-Wide-0.')])
        records = repo.alias_fold_pass()
        assert records
        record = records[0]
        assert record['reason'] == af.REASON_DIGEST_OVER_WIDTH, record
        assert record['width']['fits'] is False
        assert record['width']['single_mount_claimable'] is False
        assert record['width']['digest_tokens'] > record['width']['arena_width']
        # The existing width guard repaired it into mountable children.
        assert record['width_guard_split'], record
        for child in record['width_guard_split']:
            assert int(repo.arena.grafts[child]['ntok']) <= int(
                repo.arena.width)
    finally:
        repo.close()


# ------------------------------------------------------- non-alias controls

def test_non_alias_folds_are_byte_identical_with_the_flag_on(
        tmp_path, monkeypatch):
    """Ordinary (non-alias) folding must not change when A1 is ON."""
    records = {}
    for label, flag in (('off', '0'), ('on', '1')):
        repo = build(tmp_path, monkeypatch, flag, name='nonalias-' + label)
        try:
            ids = seed(repo, [
                harmony('The current C7-Plain-0 value is Basalt-811.'),
                harmony('The current C7-Plain-1 value is Basalt-812.'),
            ], no_fold=False)
            result = repo.arena.consolidate(ids)
            records[label] = {
                'result': result,
                'attempts': repo.arena.last_consolidation_attempts,
                'summary': repo.arena.last_consolidation_result,
                'calls': repo.arena.m.calls,
                'nodes': [{k: g.get(k) for k in
                           ('text', 'kind', 'retired', 'sources', 'ntok')}
                          for g in repo.arena.grafts],
                'alias_history': repo.alias_fold_history,
            }
        finally:
            repo.close()
    assert json.dumps(records['off'], sort_keys=True, default=str) == \
        json.dumps(records['on'], sort_keys=True, default=str)
    assert records['on']['alias_history'] == []


def test_non_alias_text_yields_no_alias_job(tmp_path, monkeypatch):
    repo = build(tmp_path, monkeypatch, '1', name='noalias')
    try:
        seed(repo, [harmony('The current C7-Plain-0 value is Basalt-811.'),
                    harmony('I want the arrival to feel welcoming.')])
        assert repo._alias_edges() == {}
        assert repo.alias_fold_pass() == []
        assert all(not g.get('retired') for g in repo.arena.grafts)
    finally:
        repo.close()


def test_two_alias_relations_in_one_node_is_not_a_clean_edge():
    text = harmony('A-1 is an alias for B-1. C-1 is an alias for D-1.')
    assert af.parse_alias_edge(text) is None


def test_alias_scan_reads_user_content_only():
    """An assistant acknowledgment is not an alias declaration."""
    node = (SYSTEM + '<|start|>user<|message|>What did we decide?<|end|>'
            + '<|start|>assistant<|channel|>final<|message|>'
            + 'A-1 is an alias for B-1.<|end|>')
    assert af.parse_alias_edge(node) is None
    assert af.parse_alias_edge(harmony('A-1 is an alias for B-1.')) == (
        'A-1', 'B-1')


# ------------------------------------------------------ deposit-path wiring

def test_deposit_funnel_merges_a_newly_written_alias(tmp_path, monkeypatch):
    """The runtime's deposit funnel runs the merge, not just the sweep."""
    repo = build(tmp_path, monkeypatch, '1', name='deposit')
    try:
        seed(repo, [harmony('The current C7-Dep-0 value is Jasper-800.')])
        before = repo._snapshot_state()
        idx = repo.arena.deposit(
            harmony('C7-DepSig-0 is an alias for C7-Dep-0.'))
        repo.arena.grafts[idx]['kind'] = 'turn'
        repo._sync_lifecycle()
        repo.runtime._guard_deposit_width(before)
        merged = [d for d in repo.alias_fold_history
                  if d['reason'] == af.REASON_MERGED]
        assert merged, repo.alias_fold_history
        assert repo.arena.grafts[idx]['retired'] is True
        assert af.names_present(
            repo.arena.grafts[merged[-1]['digest']]['text'],
            'C7-DepSig-0', 'Jasper-800')
    finally:
        repo.close()


def test_no_fold_nodes_are_still_reachable_by_the_alias_treatment(
        tmp_path, monkeypatch):
    """RD2's edges and bases all carry no_fold; the treatment must reach them."""
    repo = build(tmp_path, monkeypatch, '1', name='nofold')
    try:
        base, edge = seed(
            repo, [harmony('The current C7-NF-0 value is Jasper-900.'),
                   harmony('C7-NFSig-0 is an alias for C7-NF-0.')],
            no_fold=True)
        assert repo.arena.grafts[edge]['no_fold'] is True
        # The ORDINARY window fold cannot see them ...
        assert edge not in repo._foldable(('turn',))
        assert base not in repo._foldable(('turn',))
        # ... but the targeted alias pair job can.
        assert [m for m in repo.alias_fold_pass()
                if m['reason'] == af.REASON_MERGED]
    finally:
        repo.close()


def test_scale_many_aliases_one_pass_bounded(tmp_path, monkeypatch):
    """LT1-scale sanity: 200 nodes, 10 aliases, one bounded pass.

    Guards the two ways the sweep could misbehave at length: a re-merge loop
    (the idempotence rail and the abort exemption both have to hold) and a
    quadratic blow-up in the shared-base dependency scan.  Asserts BOUNDS,
    not wall-clock, so it cannot flake on a loaded machine: exactly one
    decision per alias, and a steady state that produces no further jobs.
    """
    repo = build(tmp_path, monkeypatch, '1', name='scale')
    try:
        texts = [harmony(f"Entity-{i}'s reserve will be {i * 7} units.")
                 for i in range(100)]
        texts += [harmony(f'Sig-{i} is an alias for Entity-{i}.')
                  for i in range(10)]
        texts += [harmony(f'We discussed the layout of section {i} today.')
                  for i in range(90)]
        seed(repo, texts)
        assert len(repo.arena.grafts) == 200
        records = repo.alias_fold_pass()
        merged = [r for r in records if r['reason'] == af.REASON_MERGED]
        assert len(merged) == 10, records
        # Exactly one decision per alias: no pair was retried.
        assert len(records) == 10
        # Every digest is single-mount claimable.
        for record in merged:
            assert record['width']['fits'] is True
            assert record['lineage'] == af.LINEAGE_EDGE_AND_BASE
        # Steady state: the sweep is idempotent and produces no new nodes.
        count = len(repo.arena.grafts)
        assert repo._alias_fold_jobs() == ()
        assert repo.alias_fold_pass() == []
        assert len(repo.arena.grafts) == count
    finally:
        repo.close()
