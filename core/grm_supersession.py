"""Production resolution for GRM supersession mount filtering.

GRM-SUP-L2-ON (operator decision, 2026-08-30) makes revision-aware
mount resolution the permanent default.  ``GRM_SUP_RESOLVE=0`` restores the
legacy pass-through path.  Callers that must freeze an experimental frame may
pass an explicit boolean; explicit values take precedence over the env.
"""

from __future__ import annotations

from collections.abc import Mapping
import os


ENV_NAME = "GRM_SUP_RESOLVE"
_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))


def env_sup_resolve_override(
    environ: Mapping[str, str] | None = None,
) -> bool | None:
    """Return the explicit env choice, or ``None`` when it is unset.

    Unknown tokens fail closed to the legacy OFF path.  This matches the
    probe-ladder default-on precedent: a malformed operator escape never
    silently enables a behavior the operator may have meant to disable.
    """
    env = os.environ if environ is None else environ
    if ENV_NAME not in env:
        return None
    value = str(env.get(ENV_NAME, "")).strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return False


def sup_resolve_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve L2: explicit caller > env escape/override > default ON."""
    if explicit is not None:
        return bool(explicit)
    override = env_sup_resolve_override(environ)
    return True if override is None else override


def sup_resolve_cli_argv(enabled: bool) -> list[str]:
    """Freeze a resolved L2 value across the e2e driver's restart exec."""
    return ["--sup-resolve"] if enabled else ["--no-sup-resolve"]
