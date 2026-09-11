#!/usr/bin/env python3
"""GRM-P1 fake-model transcript smoke — PLUMBING, never answer quality.

Runs ``scripts/grm_chat.py --fake-model --transcript`` over the registered
20-turn smoke transcript and checks the things a product surface must get
right no matter what the model says:

  * mounts happen (at least one turn mounted a routed graft);
  * one ``grm.route_receipt.v1`` per turn is in ``REPO/session_ledger.jsonl``;
  * the restart retained the repository (node count before == after) and the
    post-restart turns still route;
  * NO LIVE HISTORY LEAK — every live prompt the model saw is exactly
    ``harmony_turn(that turn's user text, None)``, so no prior turn's text
    was fed live.

The value-span scorer (``scripts.grm_lt1.score``, the C5 arm-S contiguous
ordered span rule) is applied to the recall turns and REPORTED.  It is not
gated: the fake model is a regex that copies a visible value, and its answer
rate is a statement about the stub, not about GRM.  Quality lives in the
registered GPU smoke (``artifacts/grm_p1/lead_commands.txt``).

Prior art
---------
``grm_lt1_cpu.run`` / ``grm_c7_diagnose`` (GRM contributors, 2026): a CPU
author gate that runs the real repository/admission/ladder against a stub
reader, explicitly not a quality measurement.  Taken: that discipline and
the scorer.  Ours: the leak assertion and the ledger/restart checks.  No
prior art known to me for this exact gate composition.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSCRIPT = ROOT / "fixtures/grm_p1/smoke_transcript.txt"
OUT = ROOT / "artifacts/grm_p1"

#: Recall turns and the value each should surface. ``turn`` is the LEDGER
#: turn number: ``/restart`` and ``/status`` are session events, not chat
#: turns, so they do not advance the counter — the transcript's 20 lines
#: become 19 ledger turns.  ``distance`` is how many turns back the value
#: currently in force was stated.
RECALLS = (
    {"turn": 6, "expected": "Auric-4-Alpha", "distance": 5,
     "klass": "fresh"},
    {"turn": 11, "expected": "Kestrel-9-Tango", "distance": 3,
     "klass": "correction"},
    {"turn": 14, "expected": "Gold-7-Foxtrot", "distance": 7,
     "klass": "fresh"},
    {"turn": 15, "expected": "Zenith-2-Echo", "distance": 2,
     "klass": "correction"},
    {"turn": 16, "expected": "Nadir-1-Delta", "distance": 14,
     "klass": "fresh"},
    {"turn": 17, "expected": "Vortex-3-Sierra", "distance": 14,
     "klass": "fresh"},
    {"turn": 18, "expected": "Nadir-1-Delta", "distance": 6,
     "klass": "alias"},
)


def _leak_probe(session, seen: list[dict[str, Any]]):
    """Record every CHAT-TURN live prompt the model is asked to read.

    The choke point is ``ArenaCache._attempt``:
    ``prompt_ids = self.encode(self._format_step_prompt(user_text))``.
    Wrapping ``_format_step_prompt`` on the instance captures exactly the
    text that becomes live tokens for a served turn — grafts arrive as K/V
    and never pass through here, which is the whole point.

    NOT every model call is a chat turn.  ``repo._librarian()`` runs
    CONSOLIDATION (``ArenaCache.DIGEST_PROMPTS`` / ``ERA_PROMPTS``, built by
    ``_consolidation_prompts``), which folds several grafts into one digest
    node and therefore quotes their text into its own prompt on purpose.
    That is a separate GRM operation with its own prompt shape, not the chat
    log entering the model's context, so it is recorded and EXCLUDED here
    rather than silently widening the chat-turn rule.  The exclusion is
    keyed on the consolidation prompt's own first line, so a chat turn can
    never be mistaken for a fold (a user would have to type that sentence
    verbatim, and the rule below would still check it against every other
    turn's text).
    """
    arena = session.repo.arena
    original = arena._format_step_prompt
    markers = tuple(
        p.split("\n", 1)[0].removeprefix("User: ")
        for p in tuple(getattr(arena, "DIGEST_PROMPTS", ()))
        + tuple(getattr(arena, "ERA_PROMPTS", ())))

    def traced(user_text):
        text = original(user_text)
        kind = ("consolidation"
                if any(m and m in str(user_text) for m in markers)
                else "chat_turn")
        seen.append({"user_text": str(user_text), "prompt": str(text),
                     "kind": kind})
        return text

    arena._format_step_prompt = traced
    return original


def assert_no_live_history(seen: list[dict[str, Any]],
                           user_turns: list[str]) -> dict[str, Any]:
    """THE LEAK ASSERTION.

    For every CHAT-TURN live prompt: it must equal ``harmony_turn(its own
    user text, None)`` exactly, and it must not contain any OTHER turn's
    user text.  Both halves matter — the first forbids any extra text at
    all, the second names what we are actually afraid of.

    Consolidation prompts (``kind == "consolidation"``) are counted and
    reported separately: a librarian fold quotes its source grafts by
    design.  A row with no ``kind`` is treated as a chat turn, so the
    strict rule is the default and an un-probed caller cannot opt out of
    it by accident.
    """
    from scripts.grm_e2e_session import harmony_turn

    checked = 0
    folds = 0
    for row in seen:
        if row.get("kind") == "consolidation":
            folds += 1
            continue
        prompt, user = row["prompt"], row["user_text"]
        expected = harmony_turn(user, None)
        if prompt != expected:
            raise AssertionError(
                "LIVE_HISTORY_LEAK: live prompt is not the current turn "
                f"alone.\n  expected: {expected!r}\n  got:      {prompt!r}")
        for other in user_turns:
            if other != user and other and other in prompt:
                raise AssertionError(
                    "LIVE_HISTORY_LEAK: prior turn text found in the live "
                    f"prompt.\n  prior turn: {other!r}\n  prompt: {prompt!r}")
        checked += 1
    if not checked:
        raise AssertionError("LEAK_TEST_SAW_NO_PROMPTS: nothing was served")
    return {"prompts_checked": checked,
            "rule": "prompt == harmony_turn(this turn's user text, None)",
            "consolidation_prompts_excluded": folds,
            "consolidation_note": (
                "librarian folds quote their source grafts by design; they "
                "are a separate operation from a served chat turn"),
            "prior_turn_texts_checked": len(user_turns)}


def _user_turns(transcript: Path) -> list[str]:
    """Transcript lines as a 1-based turn list (commands become '')."""
    out = []
    for raw in Path(transcript).read_text().splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        out.append("" if text.startswith("/") else text)
    return out


@contextlib.contextmanager
def pinned_environment(env: dict[str, str]):
    """Pin the profile's env for the duration, then put it back EXACTLY.

    ``grm_chat.main`` pins the environment process-wide, which is right for
    a real session: the process exists to serve one profile.  This function
    runs IN-PROCESS under pytest alongside other suites, so leaving
    ``GRM_ADMISSION_RULE=margin_first`` behind silently re-rules every test
    that runs after it — which is exactly what it did (test_grm_scout_fix4
    and test_grm_admission went RED only when run after this smoke, and
    passed alone). Restoring is not tidiness; a gate that changes another
    gate's result is not a gate.
    """
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


#: The registered arms. ``r3-A`` is the r2 configuration; ``r3-A+`` adds
#: A1's alias fold-merge (``core/grm_alias_fold.py``, default OFF). Both
#: gate on plumbing only — see the module docstring.
ARMS = {
    "A": (),
    "A+": ("GRM_ALIAS_FOLD_MERGE",),
}


def run(repo_dir: Path, *, transcript: Path = TRANSCRIPT,
        profile: str = "eb1_c2",
        pinned: tuple[str, ...] | None = None,
        arm: str | None = None) -> dict[str, Any]:
    """Run one arm. ``arm`` names a registered pin set; ``pinned`` overrides."""
    from scripts.grm_profile import resolve_profile
    if pinned is None:
        pinned = ARMS[arm] if arm is not None else ()
    resolved = resolve_profile(selection=profile, pinned=list(pinned))
    with pinned_environment(resolved["env"]):
        result = _run_pinned(repo_dir, transcript, resolved)
    result["arm"] = arm
    result["pinned"] = list(pinned)
    result["named_flags_applied"] = resolved["named_flags_applied"]
    return result


def _run_pinned(repo_dir: Path, transcript: Path,
                resolved: dict[str, Any]) -> dict[str, Any]:
    from scripts import grm_chat
    from scripts import grm_lt1 as lt

    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    session = grm_chat.ChatSession(repo_dir, resolved, fake=True)
    session.open()
    seen: list[dict[str, Any]] = []
    _leak_probe(session, seen)
    # ``/restart`` builds a NEW arena, so re-install the probe whenever the
    # session reopens; otherwise the post-restart prompts go untraced and
    # the leak assertion would silently cover only half the run.
    original_restart = session.restart

    def traced_restart():
        record = original_restart()
        _leak_probe(session, seen)
        return record

    session.restart = traced_restart
    try:
        result = grm_chat.run_transcript(session, transcript)
    finally:
        grm_chat._save_fake_codec(session)
        session.close()

    ledger = [json.loads(line) for line in
              (repo_dir / grm_chat.LEDGER_NAME).read_text().splitlines()
              if line.strip()]
    turns = [r for r in ledger if r.get("schema") == "grm.chat_turn.v1"]
    restarts = [r for r in ledger if r.get("kind") == "restart"]
    restart_events = [e for e in result["events"]
                      if e.get("command") == "/restart"]

    leak = assert_no_live_history(seen, _user_turns(transcript))

    # A turn that raised carries answer=None and an `error` field. It is
    # evidence, not a reason to stop scoring — and it must never be counted
    # as a receipt or a correct answer.
    failed = [{"turn": r["turn"], "user": r["user"], "error": r["error"]}
              for r in turns if r.get("error")]
    receipts_ok = all(
        (r.get("route_receipt") or {}).get("schema") == "grm.route_receipt.v1"
        for r in turns if not r.get("error"))
    mounted_turns = [r["turn"] for r in turns if r["mounted_ids"]]
    by_turn = {r["turn"]: r for r in turns}
    scored = []
    for spec in RECALLS:
        row = by_turn.get(spec["turn"])
        if row is None:
            scored.append({**spec, "answer": None, "mounted_ids": None,
                           "correct": False, "category": "MISSING_ROW"})
            continue
        if row.get("error"):
            scored.append({**spec, "answer": None, "mounted_ids": [],
                           "correct": False, "category": "TURN_FAILED",
                           "error": row["error"]})
            continue
        verdict = lt.score(row["answer"], spec["expected"])
        scored.append({**spec, "answer": row["answer"],
                       "mounted_ids": row["mounted_ids"],
                       "correct": bool(verdict["exact_correct"]),
                       "category": verdict["category"]})

    # The restart lands after ledger turn 15 (it is an event, not a turn).
    post_restart_mounted = [r["turn"] for r in turns
                            if r["turn"] > 15 and r["mounted_ids"]]
    recap_rows = [r for r in turns if r["user"] == grm_chat.RECAP_QUESTION]

    checks = {
        "turns_recorded": len(turns) == 19,
        "no_failed_turns": not failed,
        "route_receipt_schema_every_turn": receipts_ok,
        "mounts_happen": bool(mounted_turns),
        "restart_recorded": len(restarts) == 1,
        "restart_retained": bool(restarts) and all(
            r["retained"] for r in restarts),
        "post_restart_routes": bool(post_restart_mounted),
        "no_live_history_leak": True,
        "recap_present": bool(recap_rows),
        "recap_not_deposited": all(
            r["deposited_node_id"] is None for r in recap_rows),
    }
    return {
        "status": "PASS" if all(checks.values()) else "RED",
        "evidence_class": "CPU fake model; PLUMBING only, not answer quality",
        "profile": resolved["profile"],
        "admission_rule": resolved["admission_rule"],
        "checks": checks,
        "failed_turns": failed,
        "leak_test": leak,
        "turns": len(turns),
        "mounted_turns": mounted_turns,
        "post_restart_mounted": post_restart_mounted,
        "restart": restarts,
        "restart_events": restart_events,
        "value_span_scores": scored,
        "value_span_correct": sum(s["correct"] for s in scored),
        "value_span_of": len(scored),
        "repo": str(repo_dir),
        "ledger": str(repo_dir / grm_chat.LEDGER_NAME),
        "transcript": str(transcript),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", type=Path, default=None,
                   help="session directory (default: per-arm under OUT)")
    p.add_argument("--transcript", type=Path, default=TRANSCRIPT)
    p.add_argument("--profile", default="eb1_c2")
    p.add_argument("--arm", choices=sorted(ARMS) + ["all"], default="all",
                   help="A = as r2; A+ = A1 alias fold-merge pinned")
    p.add_argument("--receipt", type=Path, default=None,
                   help="receipt path (default: per-arm under OUT)")
    args = p.parse_args(argv)

    arms = sorted(ARMS) if args.arm == "all" else [args.arm]
    results = {}
    for arm in arms:
        slug = arm.replace("+", "plus")
        repo = args.repo or OUT / f"smoke_session_{slug}"
        result = run(repo, transcript=args.transcript, profile=args.profile,
                     arm=arm)
        receipt = args.receipt or OUT / f"smoke_receipt_{slug}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      default=str) + "\n")
        result["receipt"] = str(receipt)
        results[arm] = result
        print(json.dumps(result, indent=2, sort_keys=True, default=str))

    print("\n=== ARM SUMMARY ===")
    for arm, result in results.items():
        flags = ", ".join(result["named_flags_applied"]) or "(none)"
        print(f"  {arm:3} {result['status']:4} "
              f"turns={result['turns']:2} "
              f"value_span={result['value_span_correct']}/"
              f"{result['value_span_of']} "
              f"failed={len(result['failed_turns'])} pinned={flags}")
    return 0 if all(r["status"] == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
