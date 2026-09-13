#!/usr/bin/env python3
"""GRM-P1 — ``GRM_PROFILE``: one switch that selects a registered profile.

``eb1_c2`` selects the C2 registered profile entry
(``config/grm_eb1_profile_registered.json``, profile_id
``gpt-oss-20b-eb1-w96-live-rt1``): arena width 96, capture pin ``live``,
seat-near-live ON, RT1 rule ON, plus ``GRM_ADMISSION_RULE=margin_first``
(SCOUT-FIX-6, the rule the LT1 200-turn run used).

GRM-D2 (2026-09-11): ``eb1_c2`` IS THE SHIPPED DEFAULT.  Order
``orders/GRM_D2_DEFAULTS.md``, flip 2.  Before D2 this module's headline
was "UNSET IS TODAY'S BEHAVIOUR" and unset returned the registry's
``default_flags``; that is now the NAMED profile ``legacy_256`` and unset
returns C2.

EVIDENCE FOR THE FLIP: C2 beat the shipped defaults on every battery and
survived a restart, and every LT1/LT1.1 run -- including the r3 campaign
this order's other flips rest on -- was taken under this profile.  The
old defaults were never the measured configuration; they were the
un-measured one.

``legacy_256`` remains selectable (``GRM_PROFILE=legacy_256``), and
``GRM_LEGACY_DEFAULTS=1`` selects it for you along with the other five
round-2 rollbacks.  ``defaults`` is kept as an ALIAS for "whatever is
shipped", so a caller that asked for the defaults by name keeps getting
the defaults rather than silently getting the old ones.

Prior art
---------
``scripts.grm_c2_profile.select_profile`` and ``scripts.grm_c2_cells
.environment`` (GRM contributors, 2026) do the actual selection and the
actual env construction; both are called here unchanged.  ``core
.grm_admission.admission_rule`` owns the margin-first resolution.  Taken:
the registry, the explicit-selection-only discipline (an unknown profile
name is an error, never a silent fallback), and the "strip ambient GRM_*
before pinning" rule.  Ours (new here): the single human-facing
``GRM_PROFILE`` name, the named-flag pin mechanism with an absent-flag
receipt, and the printable resolved-flag record.  No new routing,
admission or selection algorithm.  Named-profile configuration selection is
ordinary practice (feature flags, Django settings modules, Rails
environments); no prior art known to me for this exact resolver.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from core import grm_legacy_defaults as legacy_defaults

ROOT = Path(__file__).resolve().parents[1]

PROFILE_ENV = "GRM_PROFILE"

#: Human-facing profile names -> the registered profile_id they select.
#: ``None`` means the registry's own ``default_flags`` (width 256, capture
#: pin off, seat-near-live off) -- the PRE-D2 shipped set, which since
#: GRM-D2 is a NAMED profile rather than the unnamed fallback it used to be.
PROFILES: dict[str, str | None] = {
    "legacy_256": None,
    "eb1_c2": "gpt-oss-20b-eb1-w96-live-rt1",
}

#: The profile an unset ``GRM_PROFILE`` resolves to, under the
#: ``GRM_LEGACY_DEFAULTS`` umbrella's control.  Kept as a function rather
#: than a constant because the umbrella is read at CALL time, exactly as
#: ``core.grm_admission.admission_rule`` is.
DEFAULT_PROFILE_ALIAS = "defaults"


def default_profile(environ: Mapping[str, str] | None = None) -> str:
    """The profile name an unset ``GRM_PROFILE`` selects.

    ``eb1_c2`` since GRM-D2; ``legacy_256`` under ``GRM_LEGACY_DEFAULTS=1``.
    """
    return str(legacy_defaults.default_for(PROFILE_ENV, environ))

#: Flags that are part of the profile but are not expressed as an
#: environment variable by ``grm_c2_cells.environment``; they reach the
#: arena through the CLI/constructor instead. Listed so ``/status`` and the
#: start-up printout do not imply an env var that is not actually set.
NON_ENV_FLAGS = ("arena_width", "topk", "ngen", "max_trips", "live_turns",
                 "max_live", "turn_pipeline", "vram_budget_mb", "ephemeral")

#: Optional named flags a lead may pin.  Whether a flag is EFFECTIVE is
#: decided at resolve time by ``_flag_is_read`` — never by this list — so a
#: flag that lands on the tree later starts working with no edit here, and
#: one that is removed stops claiming to.
#:
#: ``GRM_ALIAS_FOLD_MERGE`` is the alias fold-merge mechanism from the C7 r3
#: alias design memo.  It landed with A1 (``core/grm_alias_fold.py``, hooks
#: in ``core/graft_repository.py`` and ``core/grm_runtime.py``) and is
#: DEFAULT OFF there: ``env_alias_fold_override`` returns ``None`` when the
#: variable is unset, and unknown tokens fail CLOSED to OFF.  Pinning it
#: sets ``=1``, one of A1's true tokens.  Before A1 this same code reported
#: it as absent-and-not-effective; that transition needed no change here.
KNOWN_NAMED_FLAGS = ("GRM_ALIAS_FOLD_MERGE",)


def _registry() -> Mapping[str, Any]:
    from scripts.grm_c2_profile import REGISTRY, read
    return read(REGISTRY)


def _flag_is_read(name: str) -> bool:
    """True when some module on this tree actually reads ``name``.

    A pinned flag nothing reads is a no-op, and saying so is the honest
    receipt.  Only ``core/`` and ``scripts/`` are scanned: those are the
    code paths a session executes.
    """
    # The P1 surface itself names the flag (help text, this table); a
    # mention there is not a reader, so exclude our own files or every
    # pinned flag would look present.
    mine = {Path(__file__).name, "grm_chat.py"}
    for directory in ("core", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            if path.name in mine:
                continue
            try:
                if name in path.read_text():
                    return True
            except OSError:                                 # pragma: no cover
                continue
    return False


def resolve_profile(selection: str | None = None,
                    pinned: Iterable[str] | None = None,
                    environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the profile switch into flags, env and a printable record.

    ``selection`` overrides ``GRM_PROFILE``.  Unknown names raise, because a
    profile that silently falls back to defaults is a profile you cannot
    trust a receipt from (``grm_c2_profile.select_profile`` takes the same
    position on the registry's profile_id).
    """
    env_in = os.environ if environ is None else environ
    from scripts.grm_c2_cells import environment
    from scripts.grm_c2_profile import select_profile

    raw = selection if selection is not None else env_in.get(PROFILE_ENV)
    name = (str(raw).strip().lower() if raw else default_profile(env_in))
    # ``defaults`` is an ALIAS for "whatever is shipped", not a frozen
    # snapshot of the pre-D2 set: a caller asking for the defaults by name
    # must not be quietly handed last release's.
    if name == DEFAULT_PROFILE_ALIAS:
        name = default_profile(env_in)
    if selection is not None:
        source = "--profile"
    elif raw:
        source = f"{PROFILE_ENV}={raw}"
    elif legacy_defaults.legacy_defaults_enabled(env_in):
        source = (f"{PROFILE_ENV} unset, "
                  f"{legacy_defaults.ENV_NAME}=1 -> {name}")
    else:
        source = f"{PROFILE_ENV} unset -> {name} (shipped default)"
    if name not in PROFILES:
        known = ", ".join(sorted(PROFILES) + [DEFAULT_PROFILE_ALIAS])
        raise ValueError(f"unknown {PROFILE_ENV}: {raw!r} (known: {known})")

    registry = _registry()
    flags = select_profile(registry["default_flags"], PROFILES[name],
                           registry=registry)

    # ``environment`` strips every ambient GRM_* variable before pinning the
    # frame — the same discipline a C2/LT1 worker runs under, so an operator
    # cannot half-apply a profile by leaving a stale switch in their shell.
    env = {k: v for k, v in environment(flags).items()
           if k.startswith("GRM_")}

    # The admission rule is a profile property, not an arena flag: the C2
    # registry does not carry it, and LT1 registered margin_first as the
    # rule its receipts were taken under.  GRM-D2: the rule is now PINNED
    # EXPLICITLY on BOTH sides rather than only on the C2 side.  Before D2,
    # ``legacy_256``'s predecessor could POP the variable and let the
    # resolver's unset default supply ``all_tokens_bind``; after D2 the
    # unset default is ``margin_first``, so popping it would put a
    # margin-first rule inside a legacy frame -- a half-applied profile,
    # the exact failure the ambient strip above exists to prevent.
    admission_rule = "margin_first" if PROFILES[name] else "all_tokens_bind"
    env["GRM_ADMISSION_RULE"] = admission_rule

    # ``environment`` stripped EVERY ambient GRM_* key, ``GRM_LEGACY_DEFAULTS``
    # included.  That is right for the frame flags -- a stale switch must not
    # half-apply a profile -- but the umbrella is not a frame flag: it is the
    # operator's rollback, and a rollback that a profile resolution silently
    # cancels is not a rollback.  Carry it through verbatim when it was set,
    # so the four round-2 flag flips resolve the same way inside a resolved
    # profile as outside one.  The four flags themselves are deliberately NOT
    # pinned here: they are independent switches with their own env escapes
    # and their own ``--pin-flag`` channel, and a profile that pinned them
    # would make ``--pin-flag GRM_ALIAS_FOLD_MERGE`` un-overridable.
    if legacy_defaults.ENV_NAME in env_in:
        env[legacy_defaults.ENV_NAME] = env_in[legacy_defaults.ENV_NAME]
    env[PROFILE_ENV] = name

    notes: list[str] = []
    applied: list[str] = []
    for item in list(pinned or ()):
        flag, _, value = str(item).partition("=")
        flag = flag.strip()
        if not flag:
            continue
        value = value or "1"
        if not _flag_is_read(flag):
            notes.append(
                f"{flag} pinned but ABSENT on this tree (no reader in core/ "
                f"or scripts/) — recorded, not effective")
            continue
        env[flag] = value
        applied.append(f"{flag}={value}")

    return {
        "profile": name,
        "profile_id": PROFILES[name],
        "profile_source": source,
        "flags": flags,
        "env": env,
        "admission_rule": admission_rule,
        "named_flags_applied": applied,
        "named_flags_known": list(KNOWN_NAMED_FLAGS),
        "non_env_flags": list(NON_ENV_FLAGS),
        "notes": notes,
        "registry": str(ROOT / "config" / "grm_eb1_profile_registered.json"),
    }


def describe_profile(name: str) -> dict[str, Any]:
    """The flag delta a profile applies, without touching the environment."""
    registry = _registry()
    base = dict(registry["default_flags"])
    resolved = resolve_profile(selection=name, environ={})
    return {
        "profile": name,
        "changes": {k: {"default": base.get(k), "profile": v}
                    for k, v in resolved["flags"].items() if base.get(k) != v},
        "admission_rule": resolved["admission_rule"],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(
        description=f"resolve the {PROFILE_ENV} switch")
    p.add_argument("--profile", default=None)
    p.add_argument("--pin-flag", action="append", default=[])
    p.add_argument("--describe", action="store_true")
    args = p.parse_args(argv)
    if args.describe:
        print(json.dumps(
            describe_profile(args.profile or default_profile()),
            indent=2, sort_keys=True))
        return 0
    print(json.dumps(resolve_profile(selection=args.profile,
                                     pinned=args.pin_flag),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
