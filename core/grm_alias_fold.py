"""GRM-A1 — alias resolution by FOLD-MERGE, flag-gated.

THE DEFECT THIS ADDRESSES (RD2, measured, not inferred).
``/mnt/ForgeRealm/wt/grm-rd1/artifacts/grm_rd2/REPORT.md`` records 8/8 C7
alias probes mounting ONLY the alias-EDGE node ("C7-Signal-0 is an alias for
C7-AliasBase-0.", 49 seats) and never its value-bearing BASE ("The current
C7-AliasBase-0 value is Jasper-711. …", 61 seats).  Two independent reasons,
both receipted there:

  1. ``core/grm_admission.py::is_identifier_binding`` accepts rare-identifier
     membership in the candidate's OWN text.  The question carries
     ``c7-signal-0``; only the edge binds.  A-DEC then takes
     ``exactly_one_identifier_decisive_rank1`` and the singleton plan is
     ``[edge]``.  Admission does not FOLLOW the edge's referenced entity.
  2. L2 (``_revision_mount_heads``) traverses explicit ``supersedes``; the
     edge node records empty ``links`` / ``sources`` / ``supersedes``, so
     there is no structural alias->base edge to traverse even in principle.

And a hard width constraint underneath both: 49 + 61 = 110 seats > the C2
arena width of 96, so simply adding the base to the rank plan cannot seat
both full payloads together under this profile.  FIX-7 (co-mount) was STOPPED
as impossible for exactly that reason.

THE TREATMENT (option (b) of
``/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md``).
Move the join to WRITE time.  Pair the alias edge with its CURRENT base as
ONE fold job; FIX-5's source enumeration lists every fact of the base plus
the alias relation; the resulting digest names alias, base and value
together, and a single digest mount can answer the Signal probe.  This module
owns the *decisions* (is this an alias edge? which base? what lineage?);
``GraftRepository`` owns the mutations.

FLAG.  ``GRM_ALIAS_FOLD_MERGE`` — DEFAULT OFF.  With the flag OFF nothing in
this module is ever called from a serving path and behaviour is
byte-identical to the pre-A1 branch (proved by
``tests/test_grm_a1_alias_fold.py::test_flag_off_byte_identical``).  An
unknown token fails CLOSED to OFF, matching the probe-ladder / L2 / A-DEC
precedent in ``grm_supersession.py`` and ``grm_admission.py``: a malformed
operator escape never silently enables a behaviour the operator may not have
meant to enable.

PRIOR ART.
  * Sarthi et al. (2024), RAPTOR (arXiv 2401.18059) — recursively summarized
    retrieval representations.  TAKEN: the principle that combining source
    information into one retrievable summary can answer a query that no
    single source answers.  NOT taken: RAPTOR's clustering, its tree
    construction, or any of its retrieval scoring.  Ours: the alias-edge ->
    current-base PAIRING rule, the coverage-gated retirement decision, and
    the width receipt.
  * Press et al. (2022), "Measuring and Narrowing the Compositionality Gap"
    (arXiv 2210.03350) — self-ask: answer the intermediate question first.
    Antecedent for the REJECTED alternative (a) two-hop read, kept here only
    to name what this module deliberately is not.  Nothing taken.
  * Gupta & Mumick (1995), "Maintenance of Materialized Views" — a merged
    digest is a materialized join, and a later correction of a base value
    must invalidate it.  TAKEN: the framing that the write-time join owes a
    maintenance rule, which is why
    ``GraftRepository._alias_extend_correction_targets`` exists at all.
    NOT taken: any specific incremental-maintenance algorithm; the
    rule here is full invalidation by explicit supersession, not a delta
    propagation.  UNVERIFIED — lead to check; search terms
    ``Gupta Mumick 1995 maintenance of materialized views``,
    ``denormalization update propagation``.
  * GRM contributors (2026), LOCAL and verified: FIX-5 source-enumerated
    consolidation and its ``max(120, 24N)`` budget; FIX-8 / SC1.1
    ``normalize_glyphs`` identifier projection; M5 / L2 explicit
    ``supersedes`` lineage and ``correct_memory``; the LSR-P2C width guard.
    TAKEN: all of these unchanged, as the verbs this module composes.  This
    module implements no new consolidation, normalization, routing or
    supersession algorithm.
  * No prior art known to me for this exact composition (alias-edge/base
    pairing under a coverage-gated retirement rule with a pre-mount width
    receipt).
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import re
from typing import Any

from core.grm_text_norm import normalize_glyphs


from core import grm_legacy_defaults as legacy_defaults

ENV_NAME = "GRM_ALIAS_FOLD_MERGE"
_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))

#: Receipt reasons.  Every alias-fold decision names exactly one of these, so
#: a reader never has to infer why a merge did or did not happen.
REASON_MERGED = "alias_merged"
REASON_BASE_MISSING = "alias_base_missing"
REASON_CYCLE = "alias_cycle"
REASON_NOT_ALIAS = "alias_not_an_edge"
REASON_SELF = "alias_self_reference"
REASON_ALREADY_MERGED = "alias_already_merged"
REASON_FOLD_ABORTED = "alias_fold_fidelity_abort"
REASON_DIGEST_OVER_WIDTH = "alias_digest_over_width"

#: Lineage outcomes (mission item 1: "record which").
LINEAGE_EDGE_AND_BASE = "digest_supersedes_edge_and_base"
LINEAGE_EDGE_ONLY = "digest_supersedes_edge_only"

#: Maximum hops when resolving alias-of-alias.  A chain longer than this in a
#: real conversation is indistinguishable from a cycle for our purposes, and
#: the terminal-base rule below refuses rather than looping.
MAX_ALIAS_HOPS = 8


def env_alias_fold_override(
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


def alias_fold_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve A1.  GRM-D2 (2026-09-11): DEFAULT ON.

    EVIDENCE FOR THE FLIP (order ``orders/GRM_D2_DEFAULTS.md``, flip 3):
    A1 ships ON **as a member of a SET** -- F1 + F2 + F5 + A1 -- never
    alone.  LT1.1 r3 moved the four together: control 36/40 -> 38/40 on the
    lead's c2 column, 33/40 -> 37/40 on the column this tree can recompute
    (``artifacts/grm_f1/REPORT.md``).  The three-without-A1 arm REGRESSED
    aliases 10/10 -> 7/10, so a SUBSET of this set is a measured regression
    and must not be shipped.  If you turn one of the four off, turn all four
    off -- that is what ``GRM_LEGACY_DEFAULTS=1`` does.

    ``=0`` remains this flag's OFF setting, and the OFF arm of every A1
    fixture still pins the pre-A1 behaviour byte-for-byte.

    PRECEDENCE (``core.grm_legacy_defaults``, rungs 1-4): an explicit
    caller value wins, then an explicitly set ``GRM_ALIAS_FOLD_MERGE``, then the
    ``GRM_LEGACY_DEFAULTS`` umbrella, then ON.

    FAIL-CLOSED DIRECTION, stated because D2 REVERSED it: an unknown token
    now falls to the SHIPPED default (ON, or OFF under the umbrella), where
    before D2 it fell to OFF.  The rule is unchanged -- a malformed escape
    never silently selects a behaviour the operator may not have meant --
    but what the shipped behaviour IS has changed.
    """
    if explicit is not None:
        return bool(explicit)
    override = env_alias_fold_override(environ)
    if override is not None:
        return override
    return bool(legacy_defaults.default_for(ENV_NAME, environ))


# ---------------------------------------------------------------- detection

#: The alias relation as the C7 fixture writes it ("X is an alias for Y.")
#: and as LT1's natural conversation writes it ("Let's call X 'Y' from now
#: on").  These are the two forms in the registered fixtures; the parser is a
#: LEXICAL PROXY over them, deliberately NOT an entity recognizer, exactly as
#: ``grm_admission.shaped_identifier_tokens`` is a lexical proxy.  A form this
#: does not match simply yields no alias job — the pre-A1 behaviour.
_ALIAS_PATTERNS = (
    # "C7-Signal-0 is an alias for C7-AliasBase-0."  (alias, base)
    re.compile(
        r"(?<![\w-])(?P<alias>[A-Za-z0-9][\w.:-]*)\s+is\s+(?:an\s+)?alias\s+"
        r"(?:for|of)\s+(?P<base>[A-Za-z0-9][\w.:-]*)",
        re.I),
    # "Let's call Kestrel 'the Hauler' from now on"  (base, alias) — note the
    # REVERSED capture order: the natural form names the BASE first.
    re.compile(
        r"(?<![\w-])call\s+(?P<base>[A-Za-z0-9][\w.:-]*)\s+"
        r"['‘’\"“”](?P<alias>[^'‘’\"“”]+)"
        r"['‘’\"“”]",
        re.I),
    # "X is also known as Y" / "X, also called Y"  (base, alias)
    re.compile(
        r"(?<![\w-])(?P<base>[A-Za-z0-9][\w.:-]*)\s*,?\s+(?:is\s+)?also\s+"
        r"(?:known\s+as|called)\s+(?P<alias>[A-Za-z0-9][\w.:-]*)",
        re.I),
)

#: Transport markers the Harmony frame wraps a stored turn in.  The alias scan
#: reads the USER content only: an assistant acknowledgment ("Recorded.") is
#: not an alias declaration, and the system frame names no entities.
_HARMONY_USER = re.compile(
    r"<\|start\|>user<\|message\|>(.*?)<\|end\|>", re.S)
_ROLE_PREFIX = re.compile(r"(?m)^(?:User|Assistant):\s*")


def alias_scan_text(text: str) -> str:
    """The span an alias declaration may legitimately appear in.

    Harmony-framed nodes contribute their user message(s) only.  A plain
    ``User:``/``Assistant:`` exchange contributes the user lines only.  Text
    with neither framing contributes itself.
    """
    raw = str(text or "")
    users = _HARMONY_USER.findall(raw)
    if users:
        return "\n".join(users)
    if _ROLE_PREFIX.search(raw):
        out = []
        role = None
        for line in raw.splitlines():
            marker = re.match(r"^(User|Assistant):\s*(.*)$", line)
            if marker:
                role = marker.group(1).casefold()
                line = marker.group(2)
            if role in (None, "user"):
                out.append(line)
        return "\n".join(out)
    return raw


def parse_alias_edge(text: str) -> tuple[str, str] | None:
    """Return ``(alias, base)`` surface forms, or ``None``.

    Only the FIRST match is honoured: a node declaring two alias relations is
    not a clean edge, and merging it would have to choose a base arbitrarily.
    Such a node yields no alias job and keeps the pre-A1 behaviour.

    THE SCAN IS FIX-8 NORMALIZED.  Measured on the real C7 r3 checkpoints
    (lead GPU run, 2026-09-11): a stored digest reading
    ``"… and C7‑Signal‑0 is an alias for C7‑AliasBase‑0."``
    — with U+2011 NON-BREAKING HYPHEN, which the model emits — was parsed as
    the alias pair ``('0', 'C7')``, because the identifier class ``[\\w.:-]``
    does not contain U+2011 and shatters the name.  That is the SAME
    shattering SC1.1/FIX-8 was minted for ("value comparison is semantics,
    not glyphs"), and this module already depends on ``normalize_glyphs``
    through ``identifier_set``; the parser simply was not using it.  A
    spurious pair is worse than a missed one: it invents an alias relation
    for a node that asserts none, and then tries to merge it.
    """
    scan = normalize_glyphs(alias_scan_text(text))
    for pattern in _ALIAS_PATTERNS:
        matches = pattern.findall(scan)
        if len(matches) != 1:
            continue
        match = pattern.search(scan)
        alias = match.group("alias").strip().strip(".,;:")
        base = match.group("base").strip().strip(".,;:")
        if not alias or not base:
            continue
        return alias, base
    return None


def identifier_set(text: str) -> frozenset[str]:
    """FIX-8-normalized identifier set for the digest coverage receipt.

    Prior art: ``grm_admission.normalized_words`` (GRM contributors, 2026) and
    the SC1.1/FIX-8 ``normalize_glyphs`` projection it is built on.  This is
    that exact token class and casefolding, reused so the digest's identifier
    set is computed by the SAME projection routing and admission use — the
    mission's "digest identifier set (FIX-8 normalized) must contain both
    names" is checkable only if it is the same set those consumers see.
    """
    out = set()
    for token in re.findall(r"[A-Za-z0-9][\w:.,\-]*",
                            normalize_glyphs(str(text or ""))):
        token = token.rstrip(".,:;").casefold()
        if token:
            out.add(token)
    return frozenset(out)


def names_present(text: str, *names: str) -> bool:
    """Every name's identifier tokens appear in ``text``'s identifier set.

    A multi-word alias ("the Hauler") contributes each of its tokens; a
    single-token alias contributes one.  Stated as subset containment so the
    check is identical for both shapes.
    """
    have = identifier_set(text)
    for name in names:
        want = identifier_set(name)
        if not want or not (want <= have):
            return False
    return True


# ----------------------------------------------------------------- lineage

def coverage_proves_base_survives(digest_text: str,
                                  base_text: str) -> tuple[bool, list[str]]:
    """Did EVERY identifier of the base survive into the digest?

    The mission's retirement rule: the base is retired ONLY when this is
    True.  This is deliberately STRICTER than the fold's own
    ``MIN_FOLD_KEEP = 0.70`` coverage bar: a 0.70-coverage digest is good
    enough to REPLACE a window of turns in the routing surface, but it is not
    good enough to DESTROY the only active record of a base fact.  Losing 30%
    of a base's identifiers and retiring the base is exactly the
    unrecoverable-fact failure the 2026-06-11 fidelity gate was minted for.

    Returns ``(proved, missing_identifiers_sorted)``.
    """
    have = identifier_set(digest_text)
    want = identifier_set(base_text)
    missing = sorted(want - have)
    return (not missing), missing


def width_receipt(encode, digest_text: str, width: int) -> dict[str, Any]:
    """Tokenize the digest against the arena width BEFORE claiming fit.

    The mission requires this measurement to precede any single-mount claim,
    because the whole point of the merge is that 49 + 61 = 110 > 96 and a
    merged record is only useful if it comes in UNDER 96.  ``fits`` False is
    not a failure of the merge — it routes the digest through the existing
    LSR-P2C width guard / split path, and the receipt says so.
    """
    tokens = int(len(encode(str(digest_text or ""))))
    width = int(width)
    return {
        "digest_tokens": tokens,
        "arena_width": width,
        "fits": bool(width > 0 and tokens <= width),
        "single_mount_claimable": bool(width > 0 and tokens <= width),
    }


def resolve_terminal_base(base_name: str,
                          edge_of: Mapping[int, tuple[str, str]],
                          base_lookup,
                          *, origin_alias: str | None = None,
                          max_hops: int = MAX_ALIAS_HOPS):
    """Follow alias-of-alias to the TERMINAL base, or refuse with a reason.

    ``edge_of`` maps a node index to its ``(alias, base)`` surface pair.
    ``base_lookup(name)`` returns the node index whose text carries ``name``'s
    value, or ``None``.

    Rules (mission item 2):
      * a name that resolves to a value-bearing node is terminal — return it;
      * a name that resolves only to ANOTHER alias edge is followed, up to
        ``max_hops``;
      * revisiting a name, or exhausting the hop budget, REFUSES with
        ``REASON_CYCLE`` — it never returns an arbitrary member of the cycle.

    Returns ``(node_index_or_None, reason, chain)``.
    """
    chain: list[str] = []
    seen: set[str] = set()
    # The ORIGIN alias is already visited by definition: the walk started
    # from its edge.  Seeding it is what makes A -> B -> A report
    # ``alias_cycle`` instead of the misleading ``alias_base_missing`` the
    # caller's own self-exclusion would otherwise produce.
    if origin_alias:
        origin_key = " ".join(sorted(identifier_set(origin_alias)))
        if origin_key:
            seen.add(origin_key)
    name = str(base_name)
    for _ in range(int(max_hops)):
        key = " ".join(sorted(identifier_set(name)))
        if key in seen:
            return None, REASON_CYCLE, chain
        seen.add(key)
        chain.append(name)
        idx = base_lookup(name)
        if idx is None:
            return None, REASON_BASE_MISSING, chain
        pair = edge_of.get(int(idx))
        if pair is None:
            return int(idx), REASON_MERGED, chain
        # The resolved node is itself an alias edge: follow its base.
        name = pair[1]
    return None, REASON_CYCLE, chain
