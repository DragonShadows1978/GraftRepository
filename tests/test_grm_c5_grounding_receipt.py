"""GRM-H2: the collectable half of the C5 grounding campaign receipt.

``tests/test_grm_c5_grounding.py`` verifies C5's frozen registration at MODULE
IMPORT time -- ``load_fixtures()`` is what produces the ``parametrize``
argument lists, so it cannot be deferred without changing what the campaign
registered.  That registration pins ABSOLUTE paths inside the campaign's own
worktree (``/mnt/ForgeRealm/wt/grm-c5/artifacts/grm_c5/fixtures.json``) and
gitignored EB1 session artifacts under
``/mnt/ForgeRealm/GraftRepository/artifacts/grm_eb1/``.  It is the
worktree-path receipt class of ``docs/TESTS_CAMPAIGN_RECEIPTS.md``: it cannot
pass on any tree but grm-c5, by construction, and the grm-c5 fork was pruned
after its merge.

That module therefore skips itself at import (``campaign_receipt_module``), so
a tree-wide run is not aborted by "Interrupted: 1 error during collection".
This file holds the receipt that a skip would otherwise swallow: one
``campaign_receipt``-marked test that calls the SAME binding check, so the
failure is deselected by default and REPRODUCED under ``-m campaign_receipt``
/ ``--campaign-receipts``, exactly like every function-level receipt.

Nothing in ``test_grm_c5_grounding.py`` was edited except the import-time
declaration; no assertion anywhere was changed.

Prior art: the module-level ``pytest.skip(..., allow_module_level=True)``
escape for imports that cannot succeed is the documented pytest idiom for
optional dependencies (pytest-dev, 2016-); taken verbatim as an idiom.
Pairing it with a separate collectable test that reproduces the very failure
the skip hides is ours -- the skip alone would convert a receipt into silence.
No external prior art known to me for that pairing.
"""
from __future__ import annotations

import pytest

REGISTRATION = ('artifacts/grm_c5/registration.json (pins absolute '
                '/mnt/ForgeRealm/wt/grm-c5/ and gitignored '
                '/mnt/ForgeRealm/GraftRepository/artifacts/grm_eb1/ inputs)')


@pytest.mark.campaign_receipt(registration=REGISTRATION)
def test_c5_registration_binds_to_its_own_worktree():
    """The binding ``test_grm_c5_grounding.py`` evaluates at import.

    On the tree C5 registered, this returns the frozen fixture rows.
    Anywhere else it raises -- and after the grm-c5 fork was pruned it cannot
    even reach the sha comparison, because the pinned path is gone.
    """
    from scripts.grm_c5_offline import load_fixtures

    assert load_fixtures()
