"""GRM-P1 author gates for the chat surface.

Prior art: the C7/LT1 CPU author-gate pattern (GRM contributors, 2026) —
run the real repository, admission and ladder against a stub reader and
gate PLUMBING, never answer quality.  Taken unchanged.  Ours: the profile
resolver cases and the live-prompt leak assertion.  No prior art known to
me for this exact test composition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_chat                                    # noqa: E402
from scripts import grm_p1_smoke                                # noqa: E402
from scripts.grm_profile import (                               # noqa: E402
    PROFILE_ENV, describe_profile, resolve_profile,
)


# -- the one switch --------------------------------------------------------

def test_profile_unset_is_todays_behaviour():
    """Shipped defaults stay the defaults: unset changes nothing."""
    resolved = resolve_profile(environ={})
    assert resolved["profile"] == "defaults"
    assert resolved["profile_id"] is None
    assert resolved["flags"]["arena_width"] == 256
    assert resolved["flags"]["capture_pin"] == "off"
    assert resolved["flags"]["seat_near_live"] is False
    assert resolved["admission_rule"] == "all_tokens_bind"
    assert "GRM_ADMISSION_RULE" not in resolved["env"]


def test_profile_eb1_c2_selects_the_registered_entry():
    resolved = resolve_profile(selection="eb1_c2")
    assert resolved["profile_id"] == "gpt-oss-20b-eb1-w96-live-rt1"
    assert resolved["flags"]["arena_width"] == 96
    assert resolved["flags"]["capture_pin"] == "live"
    assert resolved["flags"]["seat_near_live"] is True
    assert resolved["flags"]["rt1_rule"] is True
    assert resolved["admission_rule"] == "margin_first"
    env = resolved["env"]
    assert env["GRM_ADMISSION_RULE"] == "margin_first"
    assert env["GRM_CAPTURE_PIN"] == "live"
    assert env["GRM_SEAT_NEAR_LIVE"] == "1"
    assert env["GRM_RT1_RULE"] == "1"
    assert env["GRM_PERSISTENT_BOAT"] == "0"       # EB1 ephemeral boat
    assert env[PROFILE_ENV] == "eb1_c2"


def test_profile_env_switch_is_read():
    resolved = resolve_profile(environ={PROFILE_ENV: "eb1_c2"})
    assert resolved["profile"] == "eb1_c2"
    assert resolved["admission_rule"] == "margin_first"


def test_unknown_profile_is_an_error_not_a_silent_fallback():
    with pytest.raises(ValueError, match="unknown GRM_PROFILE"):
        resolve_profile(selection="not_a_profile")


def test_profile_strips_ambient_grm_switches():
    """A stale switch in the operator's shell cannot half-apply a profile."""
    resolved = resolve_profile(
        selection="eb1_c2",
        environ={"GRM_SEAT_NEAR_LIVE": "0", "GRM_CAPTURE_PIN": "off"})
    assert resolved["env"]["GRM_SEAT_NEAR_LIVE"] == "1"
    assert resolved["env"]["GRM_CAPTURE_PIN"] == "live"


def test_absent_named_flag_is_recorded_not_claimed():
    """GRM_ALIAS_FOLD_MERGE does not exist on this tree; say so."""
    resolved = resolve_profile(selection="eb1_c2",
                               pinned=["GRM_ALIAS_FOLD_MERGE"])
    assert resolved["named_flags_applied"] == []
    assert "GRM_ALIAS_FOLD_MERGE" not in resolved["env"]
    assert any("ABSENT on this tree" in n for n in resolved["notes"])


def test_present_named_flag_is_pinned():
    """A flag the tree actually reads does reach the environment."""
    resolved = resolve_profile(selection="eb1_c2",
                               pinned=["GRM_DEMAND_NGH=1"])
    assert resolved["env"]["GRM_DEMAND_NGH"] == "1"
    assert "GRM_DEMAND_NGH=1" in resolved["named_flags_applied"]


def test_describe_profile_names_the_delta():
    delta = describe_profile("eb1_c2")["changes"]
    assert delta["arena_width"] == {"default": 256, "profile": 96}
    assert delta["capture_pin"] == {"default": "off", "profile": "live"}


def test_resolved_flag_printout_names_the_frame():
    text = grm_chat.format_resolved(resolve_profile(selection="eb1_c2"))
    assert "EB1 ephemeral boat" in text
    assert "chat log NEVER in context" in text
    assert "GRM_ADMISSION_RULE=margin_first" in text
    assert "arena width      : 96" in text


# -- the CLI ---------------------------------------------------------------

def test_help_works(capsys):
    with pytest.raises(SystemExit) as exc:
        grm_chat.parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--repo" in out
    assert "--transcript" in out
    assert "GRM_PROFILE=eb1_c2" in out


def test_repo_is_required():
    with pytest.raises(SystemExit):
        grm_chat.parse_args([])


def test_print_flags_exits_clean(capsys):
    assert grm_chat.main(["--repo", "/nonexistent", "--profile", "eb1_c2",
                          "--print-flags"]) == 0
    assert "margin_first" in capsys.readouterr().out


def test_resume_without_a_repository_fails(tmp_path, capsys):
    assert grm_chat.main(["--repo", str(tmp_path / "missing"),
                          "--resume"]) == 2
    assert "no repository" in capsys.readouterr().err


# -- the leak assertion ----------------------------------------------------

def test_leak_assertion_catches_a_planted_leak():
    """The assertion must be able to FAIL — prove it on a forged prompt."""
    from scripts.grm_e2e_session import harmony_turn
    prior = "The current orion pin value is Auric-4-Alpha."
    now = "What is the current orion pin value?"
    forged = harmony_turn(prior + "\n" + now, None)
    with pytest.raises(AssertionError, match="LIVE_HISTORY_LEAK"):
        grm_p1_smoke.assert_no_live_history(
            [{"user_text": now, "prompt": forged}], [prior, now])


def test_leak_assertion_accepts_the_clean_prompt():
    from scripts.grm_e2e_session import harmony_turn
    prior = "The current orion pin value is Auric-4-Alpha."
    now = "What is the current orion pin value?"
    clean = harmony_turn(now, None)
    result = grm_p1_smoke.assert_no_live_history(
        [{"user_text": now, "prompt": clean}], [prior, now])
    assert result["prompts_checked"] == 1


def test_leak_assertion_rejects_an_empty_run():
    with pytest.raises(AssertionError, match="LEAK_TEST_SAW_NO_PROMPTS"):
        grm_p1_smoke.assert_no_live_history([], ["anything"])


def test_a_row_with_no_kind_gets_the_strict_rule():
    """The fold exclusion must be opt-IN, never the default."""
    prior = "The current orion pin value is Auric-4-Alpha."
    with pytest.raises(AssertionError, match="LIVE_HISTORY_LEAK"):
        grm_p1_smoke.assert_no_live_history(
            [{"user_text": "q", "prompt": "leaked " + prior}], [prior, "q"])


def test_consolidation_prompts_are_excluded_but_counted():
    """A librarian fold quotes its sources by design — counted, not gated.

    Folds are excluded ONLY when the probe labelled them, and the label is
    derived from the arena's own DIGEST/ERA prompt text, so a chat turn
    cannot be relabelled into the exclusion.
    """
    prior = "The current orion pin value is Auric-4-Alpha."
    now = "What is the current orion pin value?"
    from scripts.grm_e2e_session import harmony_turn
    result = grm_p1_smoke.assert_no_live_history(
        [{"user_text": now, "prompt": harmony_turn(now, None),
          "kind": "chat_turn"},
         {"user_text": "fold", "prompt": "archive: " + prior,
          "kind": "consolidation"}],
        [prior, now])
    assert result["prompts_checked"] == 1
    assert result["consolidation_prompts_excluded"] == 1


def test_fold_markers_come_from_the_arena_not_a_literal():
    """The probe's exclusion keys are the arena's real prompt constants."""
    from core.graft_arena import ArenaCache
    assert ArenaCache.DIGEST_PROMPTS
    assert any("For the archive" in p for p in ArenaCache.DIGEST_PROMPTS)


def test_live_prompt_is_the_current_turn_only():
    """The structural proof, read off the production prompt builder."""
    from core.graft_arena import ArenaCache
    from scripts.grm_e2e_session import harmony_turn

    class _Probe:
        prompt_template = staticmethod(harmony_turn)
        _format_step_prompt = ArenaCache._format_step_prompt

    text = _Probe()._format_step_prompt("what is the orion pin value?")
    assert text == harmony_turn("what is the orion pin value?", None)
    assert "Auric-4-Alpha" not in text


# -- the transcript smoke (fake model, CPU) --------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    repo = tmp_path_factory.mktemp("grm_p1_smoke") / "session"
    return grm_p1_smoke.run(repo)


def test_smoke_passes(smoke):
    assert smoke["status"] == "PASS", json.dumps(smoke["checks"], indent=2)


def test_smoke_restores_the_environment(tmp_path):
    """A gate that changes another gate's result is not a gate.

    The smoke pins ``GRM_ADMISSION_RULE=margin_first``; leaving it behind
    re-ruled every suite that ran after it (test_grm_scout_fix4 and
    test_grm_admission went RED only in combination, and passed alone).
    """
    import os
    keys = ("GRM_ADMISSION_RULE", "GRM_CAPTURE_PIN", "GRM_SEAT_NEAR_LIVE",
            "GRM_PROFILE", "GRM_PERSISTENT_BOAT")
    before = {k: os.environ.get(k) for k in keys}
    grm_p1_smoke.run(tmp_path / "session")
    assert {k: os.environ.get(k) for k in keys} == before


def test_pinned_environment_restores_on_an_exception():
    """Restoration must survive a failing run, not just a clean one."""
    import os
    key = "GRM_ADMISSION_RULE"
    before = os.environ.get(key)
    with pytest.raises(RuntimeError):
        with grm_p1_smoke.pinned_environment({key: "margin_first"}):
            assert os.environ[key] == "margin_first"
            raise RuntimeError("boom")
    assert os.environ.get(key) == before


def test_smoke_every_turn_has_a_route_receipt(smoke):
    assert smoke["checks"]["route_receipt_schema_every_turn"]
    assert smoke["turns"] == 19


def test_smoke_mounts_happen(smoke):
    assert smoke["checks"]["mounts_happen"]
    assert smoke["mounted_turns"]


def test_smoke_restart_retains_memory(smoke):
    assert smoke["checks"]["restart_retained"]
    restart = smoke["restart"][0]
    assert restart["nodes_before"] == restart["nodes_after"] > 0
    assert restart["previous_process_id"] != restart["process_id"]


def test_smoke_routes_after_the_restart(smoke):
    assert smoke["checks"]["post_restart_routes"]
    assert smoke["post_restart_mounted"]


def test_smoke_no_live_history_leak(smoke):
    assert smoke["checks"]["no_live_history_leak"]
    assert smoke["leak_test"]["prompts_checked"] >= smoke["turns"]


def test_smoke_recap_is_not_deposited(smoke):
    assert smoke["checks"]["recap_present"]
    assert smoke["checks"]["recap_not_deposited"]


def test_smoke_value_spans_are_reported_not_gated(smoke):
    """The scorer runs and produces a row per recall; the RATE is evidence
    about the stub reader, so it is reported, never asserted."""
    assert smoke["value_span_of"] == len(grm_p1_smoke.RECALLS)
    assert all("category" in row for row in smoke["value_span_scores"])


# -- session mechanics -----------------------------------------------------

def test_session_resume_continues_the_turn_counter(tmp_path):
    resolved = resolve_profile(selection="eb1_c2")
    repo = tmp_path / "session"
    first = grm_chat.ChatSession(repo, resolved, fake=True)
    first.open()
    try:
        first.ask("The current orion pin value is Auric-4-Alpha.")
        first.ask("The current lyra dock value is Nadir-1-Delta.")
        nodes = len(first.repo.arena.grafts)
    finally:
        grm_chat._save_fake_codec(first)
        first.close()

    second = grm_chat.ChatSession(repo, resolved, fake=True)
    second.open()
    try:
        assert second.turn_idx == 2          # counter recovered from ledger
        assert len(second.repo.arena.grafts) == nodes
        row = second.ask("What is the current orion pin value?")
        assert row["turn"] == 3
    finally:
        grm_chat._save_fake_codec(second)
        second.close()


def test_status_reports_the_frame_and_flags(tmp_path):
    resolved = resolve_profile(selection="eb1_c2")
    session = grm_chat.ChatSession(tmp_path / "session", resolved, fake=True)
    session.open()
    try:
        session.ask("The current orion pin value is Auric-4-Alpha.")
        status = session.status()
        assert status["ephemeral_boat"] is True
        assert status["arena_width"] == 96
        assert status["admission_rule"] == "margin_first"
        assert status["profile"] == "eb1_c2"
        assert status["repository_nodes"] >= 1
        assert "mounted_ids" in status and "seats_used" in status
    finally:
        grm_chat._save_fake_codec(session)
        session.close()


def test_ledger_rows_are_one_per_turn(tmp_path):
    resolved = resolve_profile(selection="eb1_c2")
    repo = tmp_path / "session"
    session = grm_chat.ChatSession(repo, resolved, fake=True)
    session.open()
    try:
        session.ask("The current orion pin value is Auric-4-Alpha.")
        session.ask("What is the current orion pin value?")
    finally:
        grm_chat._save_fake_codec(session)
        session.close()
    rows = [json.loads(line) for line
            in (repo / grm_chat.LEDGER_NAME).read_text().splitlines()
            if line.strip()]
    turns = [r for r in rows if r.get("schema") == "grm.chat_turn.v1"]
    assert [r["turn"] for r in turns] == [1, 2]
    for row in turns:
        receipt = row["route_receipt"]
        assert receipt["schema"] == "grm.route_receipt.v1"
        assert receipt["session_id"] == turns[0]["route_receipt"]["session_id"]
        assert "route" in receipt and "admission" in receipt


def test_open_refuses_a_geometry_mismatch(tmp_path):
    """An arena that did not come up at the profile's width is a RED open.

    Drives the real ``ChatSession.open`` with a width the arena will not
    honour (the fake arena is constructed at ``flags['arena_width']``, so
    the mismatch is forced by moving the expectation after construction).
    """
    resolved = resolve_profile(selection="eb1_c2")
    session = grm_chat.ChatSession(tmp_path / "session", resolved, fake=True)
    real_open = grm_chat._open_fake_repo

    def narrow(session_dir, flags):
        repo, info = real_open(session_dir, dict(flags, arena_width=64))
        return repo, info

    grm_chat._open_fake_repo = narrow
    try:
        with pytest.raises(ValueError, match="PROFILE_GEOMETRY_MISMATCH"):
            session.open()
    finally:
        grm_chat._open_fake_repo = real_open
        session.close()


def test_open_refuses_a_non_ephemeral_frame(tmp_path):
    """The EB1 boat is the production frame; a persistent one is refused."""
    resolved = resolve_profile(selection="eb1_c2")
    session = grm_chat.ChatSession(tmp_path / "session", resolved, fake=True)
    real_open = grm_chat._open_fake_repo

    def persistent(session_dir, flags):
        repo, info = real_open(session_dir, flags)
        repo.arena.ephemeral = False
        return repo, info

    grm_chat._open_fake_repo = persistent
    try:
        with pytest.raises(ValueError, match="FRAME_NOT_EPHEMERAL"):
            session.open()
    finally:
        grm_chat._open_fake_repo = real_open
        session.close()
