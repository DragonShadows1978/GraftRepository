"""GRM-F1 — source retention after a fold, flag-gated (DEFAULT OFF).

THE DEFECT THIS ADDRESSES (LT1.1 r2, measured, not inferred).
``artifacts/grm_d1/lt1_1/miss_causes.json`` plus
``artifacts/grm_d1/REPORT.md`` §3 record that across BOTH r2 arms a SOURCE
node appears in ``route_info.mounts`` in 0 of 57 correct rows: every answer
the campaign got right was read off a fold DIGEST or an ERA index, never off
the turn that originally carried the fact.  That is not an accident of
ranking — it is the lifecycle.  FIX-5 consolidation RETIRES a fold's sources
the moment the digest clears its ``MIN_FOLD_KEEP`` coverage bar
(``ArenaCache._deposit_consolidation``: ``grafts[i]["retired"] = True``), and
``ArenaCache._route_cand_base`` excludes every retired node from the routing
candidate base.  After a fold, the digest's own prose is the ONLY routable
record of the fact.

Coverage ≥ 0.70 is a bar on the fold's FACT SET (identifiers plus multi-word
named entities), not on the answerable content of the sources, so a digest
that passes can still have dropped the sentence that makes a fact usable.
The r2 Breakwater rows are the receipt: at d ≥ 100 three different wrong
coordinate tuples came back across the arms, each one confabulated from an
era index whose sources were gone.

THE TREATMENT.  ADD the digest instead of SUBSTITUTING it.  With
``GRM_FOLD_RETAIN_SOURCES`` ON, a fold's sources keep ``retired`` unset and
stay in ``_route_cand_base`` — they remain clean readers and clean topical
routers — while the digest is deposited exactly as before (same text, same
``sources``, same inherited ``child_cents`` and ``rare`` keys, same
``kind``).  The sources are marked ``no_fold`` so the stateless librarian
plan does not re-select them forever, and their lineage records
``digest_of`` (which digest folded me) opposite the digest's
``retained_sources`` (which sources I did NOT retire).

WHAT THIS DELIBERATELY DOES NOT TOUCH.
  * SUPERSESSION.  Retention is about COMPRESSION, never about CORRECTNESS.
    ``correct_memory`` / ``forget`` / ``_alias_retire_reassigned`` retire by
    their own explicit rule and are unchanged; a retained source that still
    carries a corrected value is retired by the correction's own text match
    exactly as an unfolded turn always was, and the digest carrying it is
    retired by the same pass (plus A1's
    ``_alias_extend_correction_targets`` entity rule when A1 is on).  Under
    retention a correction reaches MORE nodes, never fewer, because the
    source it must retire is still active to be matched.
  * GRM-A1.  ``_alias_fold_once`` makes its OWN explicit lineage decision
    after the fold returns (the edge is always retired; the base is retired
    only when ``coverage_proves_base_survives``), and it already un-retires
    the base by hand on the ``LINEAGE_EDGE_ONLY`` and ``names_present``
    paths.  Those explicit assignments run after ``_deposit_consolidation``
    either way, so A1's end state is invariant under this flag.  Retention
    must not resurrect a bare alias edge — that IS the RD2 defect A1 exists
    to fix — so ``_deposit_consolidation`` takes an explicit
    ``retain=False`` override on the A1 path.
  * The width guard, ``MIN_FOLD_KEEP``, the fold QC, the fidelity abort, the
    era rules, the route ranking and the admission ladder: untouched.

FLAG.  ``GRM_FOLD_RETAIN_SOURCES`` — DEFAULT OFF.  With the flag OFF
``retain_sources_enabled()`` returns False, ``_deposit_consolidation`` takes
its pre-F1 branch, and no other function in this module is ever called from
a serving path, so behaviour is byte-identical to the pre-F1 branch (proved
by ``tests/test_grm_f1_fold_retain.py::test_flag_off_byte_identical`` and the
C2 132-plan replay gate).  An unknown token fails CLOSED to OFF, matching the
A1 / probe-ladder / L2 / A-DEC precedent in ``grm_alias_fold.py``,
``grm_supersession.py`` and ``grm_admission.py``: a malformed operator escape
never silently enables a behaviour the operator may not have meant to enable.

PRIOR ART.
  * GRM contributors (2026), LOCAL and verified — the pieces reused
    UNCHANGED, none of which this module reimplements:
      - FIX-5 source-enumerated consolidation, its ``_fact_set`` /
        ``_coverage`` fidelity bar and ``MIN_FOLD_KEEP`` = 0.70
        (``core/graft_arena.py``).  TAKEN: the whole fold, verbatim.  This
        module changes only what happens to the sources AFTER the digest is
        accepted.
      - LSR-P2C ``_guard_deposit_width`` and its degenerate-child rejection
        (SCOUT-FIX-9), whose contract is that REJECTED sources STAY ACTIVE.
        TAKEN: the precedent that an active source alongside a derived node
        is an already-supported repository state, and its
        ``no_fold`` exemption channel, reused verbatim as the anti-reselect
        rail here.
      - A1 lineage (``core/grm_alias_fold.py``): the
        ``supersedes`` / ``superseded_by`` / ``metadata.active`` triple and
        the "which of the two happened is RECORDED" discipline.  TAKEN: the
        shape of the lineage record; ``digest_of`` / ``retained_sources``
        are the retention-side names for the same idea.  NOT taken: A1's
        alias pairing, its width receipt, or its reassignment sweep.
  * Sarthi et al. (2024), RAPTOR (arXiv 2401.18059) — recursive summary
    trees for retrieval.  RAPTOR's "collapsed tree" retrieval searches
    summary nodes AND the original leaves together, which is exactly the
    ON-state topology here; its default tree traversal searches summaries
    only, which is the OFF state.  TAKEN: the framing that leaf retention
    versus leaf substitution is a RETRIEVAL-TOPOLOGY choice with measurable
    consequences, not a storage detail.  NOT taken: RAPTOR's clustering
    (GMM/UMAP), its tree construction, its retrieval scoring, or any claim
    that collapsed-tree wins — this module measures our own.  UNVERIFIED —
    lead to check; search terms ``RAPTOR collapsed tree retrieval``,
    ``recursive abstractive summarization retrieval leaves``.
  * Gray & Reuter (1993) / log-structured storage, and Rosenblum & Ousterhout
    (1992) LFS — compaction that writes a merged record and only then frees
    the inputs, versus one that frees on write.  TAKEN: nothing mechanical;
    named because "the merged record is added, the inputs are reclaimed on a
    SEPARATE policy" is the standard framing and I am re-deriving it here,
    not inventing it.  UNVERIFIED — lead to check; search terms
    ``log-structured merge compaction retain inputs``, ``LSM tombstone vs
    live key retention``.
  * Anderson & Schooler, or any specific memory-consolidation literature
    claiming gist-plus-trace coexistence: NO prior art known to me that I
    can name precisely enough to cite.  UNVERIFIED — lead to check; search
    terms ``fuzzy-trace theory gist verbatim dual storage``,
    ``memory consolidation does not erase the trace``.
  * No prior art known to me for this exact composition (a coverage-gated
    fold whose source retirement is an independently flagged policy, with a
    ``no_fold`` anti-reselect rail and split residency accounting).
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

from core import grm_legacy_defaults as legacy_defaults

ENV_NAME = "GRM_FOLD_RETAIN_SOURCES"
_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))

#: Lineage keys.  Named so a reader of a node's metadata never has to infer
#: why an ACTIVE turn is also listed inside a digest's ``sources``.
LINEAGE_DIGEST_OF = "digest_of"
LINEAGE_RETAINED_SOURCES = "retained_sources"

#: The lineage outcome recorded on the digest, mirroring A1's
#: ``LINEAGE_EDGE_AND_BASE`` / ``LINEAGE_EDGE_ONLY`` vocabulary.
LINEAGE_SOURCES_RETIRED = "digest_supersedes_sources"
LINEAGE_SOURCES_RETAINED = "digest_added_sources_retained"


def env_retain_sources_override(
    environ: Mapping[str, str] | None = None,
) -> bool | None:
    """Return the explicit environment choice, or ``None`` when unset.

    ``None`` is ALSO what an unknown token returns, and that is the GRM-D2
    change: an unknown token used to return False (pinning OFF), which was
    fail-closed only while OFF was the shipped default.  It now declines to
    decide, so the caller falls through to the shipped default -- ON, or
    OFF under ``GRM_LEGACY_DEFAULTS=1``.  Same rule as before ("a malformed
    escape never silently selects a behaviour the operator may not have
    meant"); the thing it points at moved.
    """
    env = os.environ if environ is None else environ
    if ENV_NAME not in env:
        return None
    value = str(env.get(ENV_NAME, "")).strip().casefold()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return None


def retain_sources_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve F1.  GRM-D2 (2026-09-11): DEFAULT ON.

    EVIDENCE FOR THE FLIP (order ``orders/GRM_D2_DEFAULTS.md``, flip 3):
    F1 ships ON **as a member of a SET** -- F1 + F2 + F5 + A1 -- never
    alone.  LT1.1 r3 moved the four together: control 36/40 -> 38/40 on the
    lead's c2 column, 33/40 -> 37/40 on the column this tree can recompute
    (``artifacts/grm_f1/REPORT.md``).  The three-without-A1 arm REGRESSED
    aliases 10/10 -> 7/10, so a SUBSET of this set is a measured regression
    and must not be shipped.  If you turn one of the four off, turn all four
    off -- that is what ``GRM_LEGACY_DEFAULTS=1`` does.

    ``=0`` remains this flag's OFF setting, and the OFF arm of every F1
    fixture still pins the pre-F1 behaviour byte-for-byte.

    PRECEDENCE (``core.grm_legacy_defaults``, rungs 1-4): an explicit
    caller value wins, then an explicitly set ``GRM_FOLD_RETAIN_SOURCES``, then the
    ``GRM_LEGACY_DEFAULTS`` umbrella, then ON.

    FAIL-CLOSED DIRECTION, stated because D2 REVERSED it: an unknown token
    now falls to the SHIPPED default (ON, or OFF under the umbrella), where
    before D2 it fell to OFF.  The rule is unchanged -- a malformed escape
    never silently selects a behaviour the operator may not have meant --
    but what the shipped behaviour IS has changed.
    """
    if explicit is not None:
        return bool(explicit)
    override = env_retain_sources_override(environ)
    if override is not None:
        return override
    return bool(legacy_defaults.default_for(ENV_NAME, environ))


def digest_lineage(retained: bool, sources) -> dict[str, Any]:
    """The lineage fields a fold deposits on the DIGEST node.

    ``retained`` False reproduces the pre-F1 record exactly (``sources`` is
    the digest's own list and nothing else is added), so an OFF-flag caller
    writes byte-identical metadata.
    """
    idxs = [int(i) for i in sources]
    if not retained:
        return {}
    return {
        LINEAGE_RETAINED_SOURCES: idxs,
        "fold_lineage": LINEAGE_SOURCES_RETAINED,
    }


def source_lineage(digest_idx: int) -> dict[str, Any]:
    """The lineage fields a RETAINED source records about its digest."""
    return {LINEAGE_DIGEST_OF: int(digest_idx)}


def retained_source_indices(grafts) -> list[int]:
    """Every ACTIVE node that a fold folded but did not retire.

    Used by the residency accounting to report retained-source seats
    SEPARATELY from ordinary turn seats, so a residency bound can never be
    read as "retention costs nothing" or "retention doubled the arena"
    without the split being visible.
    """
    out = []
    for i, g in enumerate(grafts):
        if g.get("retired"):
            continue
        meta = g.get("metadata") or {}
        if meta.get(LINEAGE_DIGEST_OF) is not None:
            out.append(int(i))
    return out
