"""GRM-D2 — ``GRM_LEGACY_DEFAULTS``: the one-line rollback for round 2.

WHY THIS MODULE EXISTS
----------------------
On 2026-09-11 the lead flipped six shipped defaults at once (order
``orders/GRM_D2_DEFAULTS.md``), on David's standing grant: *"You can make
the best decision in your judgement. I can roll it back if I have to."*
A rollback the operator has to assemble out of six separate environment
variables is not a rollback he can perform under pressure, so every flip
was paired with this umbrella at the same time it landed:

``GRM_LEGACY_DEFAULTS=1`` restores EVERY pre-round-2 default at once.

WHAT IT COVERS (the complete round-2 delta; nothing else on the tree reads
this variable)

===========================================  ============  ==============
resolver                                     pre-round-2   round-2 default
===========================================  ============  ==============
``grm_admission.admission_rule``             all_tokens_bind  margin_first
``grm_profile.resolve_profile`` (profile)    ``legacy_256``   ``eb1_c2``
``grm_fold_retain.retain_sources_enabled``   OFF              ON
``grm_fold_alias_guard.fold_alias_guard_enabled``  OFF        ON
``grm_admission.route_sole_binder_insurance_enabled``  OFF    ON
``grm_alias_fold.alias_fold_enabled``        OFF              ON
===========================================  ============  ==============

PRECEDENCE, stated once and obeyed by every resolver that calls in here:

1. an EXPLICIT caller value (constructor / CLI) — always wins;
2. the flag's OWN environment variable, when the operator set it — so
   ``GRM_LEGACY_DEFAULTS=1 GRM_ALIAS_FOLD_MERGE=1`` is legacy-everything
   EXCEPT A1, which is a real and useful bisection posture;
3. ``GRM_LEGACY_DEFAULTS`` — the umbrella, which only ever changes what
   an UNSET flag resolves to;
4. the round-2 shipped default.

The umbrella is deliberately a DEFAULT SELECTOR, never an override: rule 2
above is what makes it usable for bisecting which of the six flips caused a
regression, which is the whole reason an operator reaches for it.

FAIL-CLOSED DIRECTION.  An unknown token for ``GRM_LEGACY_DEFAULTS`` fails
closed to OFF, i.e. to the round-2 defaults — the same stance every other
switch on this tree takes and for the same reason: a malformed operator
escape must never silently select a behaviour the operator may not have
meant to select, and after this order the shipped behaviour IS round 2.
Note the consequence, stated rather than left to be discovered: the four
flag flips and the admission rule now fail closed to ON / ``margin_first``,
which is the OPPOSITE direction from the one their own docstrings named
before this order, because the thing they fail closed TO is "the shipped
default", not "the older behaviour".

Prior art
---------
* The umbrella itself is the ordinary "one switch restores the previous
  release's behaviour" pattern: Django's ``DEFAULT_AUTO_FIELD`` /
  transitional settings (Django contributors, 2020-), glibc's
  ``GLIBC_TUNABLES`` and ``_FORTIFY_SOURCE`` compatibility knobs, the
  Linux kernel's ``CONFIG_COMPAT_*`` and ``sysctl`` compatibility toggles,
  and Windows' AppCompat shims.  TAKEN: the shape — group the release's
  behaviour changes behind one named rollback, and let an individually-set
  flag still win so the rollback can bisect.  NOT taken: any mechanism;
  this is 60 lines of environment resolution.  UNVERIFIED against the
  literature (no network in this sandbox) — lead to check; search terms
  ``feature flag kill switch rollback default``, ``compatibility switch
  restore previous release defaults``, ``Django transitional settings``.
* The precedence ladder (explicit caller > own env > umbrella > default)
  and the fail-closed-on-unknown-token rule are TAKEN VERBATIM from this
  tree's own resolvers — ``grm_supersession.env_sup_resolve_override``
  (L2, default ON), ``grm_admission.env_adm_decisive_override`` (A-DEC,
  default ON) and ``grm_alias_fold.env_alias_fold_override`` (A1,
  previously default OFF), GRM contributors, 2026.  Nothing about the
  ladder is new here; only the fourth rung is.
* No prior art known to me for this exact composition (a per-order
  umbrella that is a default selector rather than an override, so that it
  composes with per-flag pins for bisection).
"""

from __future__ import annotations

from collections.abc import Mapping
import os

ENV_NAME = "GRM_LEGACY_DEFAULTS"

_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))

#: The round-2 flips this umbrella reverses, as
#: ``{flag env var: (pre-round-2 default, round-2 default)}``.  Documented
#: as data so ``docs/GRM_DEFAULTS_2026-09-11.md``, ``--print-flags`` and any
#: future audit read ONE table rather than re-deriving the delta by hand.
ROUND2_FLIPS: dict[str, tuple[object, object]] = {
    "GRM_ADMISSION_RULE": ("all_tokens_bind", "margin_first"),
    "GRM_PROFILE": ("legacy_256", "eb1_c2"),
    "GRM_FOLD_RETAIN_SOURCES": (False, True),
    "GRM_FOLD_ALIAS_GUARD": (False, True),
    "GRM_ROUTE_SOLE_BINDER_INSURANCE": (False, True),
    "GRM_ALIAS_FOLD_MERGE": (False, True),
}


def legacy_defaults_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True when the operator asked for the pre-round-2 defaults.

    Unknown tokens fail CLOSED to False, i.e. to the round-2 defaults: see
    the module docstring's FAIL-CLOSED DIRECTION paragraph.
    """
    env = os.environ if environ is None else environ
    if ENV_NAME not in env:
        return False
    return str(env.get(ENV_NAME, "")).strip().casefold() in _ENV_TRUE


def default_for(flag: str, environ: Mapping[str, str] | None = None):
    """The default ``flag`` resolves to when its OWN variable is unset.

    This is rung 3 of the precedence ladder.  A caller that already found
    an explicit value (rung 1) or an explicit environment token (rung 2)
    must not reach here — those rungs win by construction.
    """
    if flag not in ROUND2_FLIPS:                       # pragma: no cover
        raise KeyError(f"{flag} is not a round-2 flip: {sorted(ROUND2_FLIPS)}")
    legacy, shipped = ROUND2_FLIPS[flag]
    return legacy if legacy_defaults_enabled(environ) else shipped
