#!/usr/bin/env python3
"""GRM-P1 — the product surface: an interactive GRM chat session.

One command, one repository directory, one model.  The chat log is NEVER in
the model's context: every user turn is served as *the question plus routed
grafts only*, and the turn's text is deposited into the repository after the
answer (David's spec, 2026-09-02; EB1 ephemeral boat).

    python3 scripts/grm_chat.py --repo ~/grm_sessions/notes

This is a SURFACE over the production ladder the C2/C7/LT1 batteries ran.
It reuses, and does not fork:

  * ``scripts.grm_e2e_session.load_model_and_repo`` — model + repository
    construction (the serving path).
  * ``scripts.grm_e2e_session._probe_ladder_chat`` — the production answer
    path (route → admission → mount → attempt), ``defer_memory=True`` so the
    answer itself is not redeposited.
  * ``scripts.grm_e2e_session.harmony_turn`` + ``arena.feed`` — the EB1
    complete-turn deposit, exactly as ``grm_lt1_worker.execute`` does it.
  * ``scripts.grm_e2e_session._turn_route_receipt`` → ``core.grm_three_pass
    .build_route_receipt`` — the ``grm.route_receipt.v1`` record, one per
    turn, appended to ``REPO/session_ledger.jsonl``.
  * ``scripts.grm_lt1.score`` — the C5 arm-S value-span scorer, used by the
    batch scorer only (never by the interactive path).
  * ``scripts.grm_c2_cells.environment`` / ``scripts.grm_c2_profile
    .select_profile`` — the registered C2 profile flags.

Prior art
---------
Everything above is local GRM prior art (GRM contributors, 2026): EB1
ephemeral frame, the RS3 seat/capture pin, RT1.1 admission isolation,
SCOUT-FIX-6 margin-first admission, the C2 registered profile and the LT1
200-turn dialogue harness.  Taken: the serving ladder, the deposit contract,
the receipt schema, the scorer, the flag registry.  Ours (new here): the
interactive REPL surface, the ``GRM_PROFILE`` single-switch resolver, the
per-session ledger file, the ``/status`` ``/recap`` ``/restart`` commands,
and the live-prompt leak assertion.  A read-eval-print loop over a retrieval
memory is not novel — the shape is that of any REPL (Python's ``code``
module, readline REPLs generally) — and no retrieval, routing, admission or
scoring algorithm is introduced by this file.  No prior art known to me for
this exact composition; unverified — lead to check, search terms:
"retrieval-augmented chat without conversation history in context",
"KV cache graft memory interactive session".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_profile import (  # noqa: E402
    PROFILE_ENV, resolve_profile,
)

LEDGER_NAME = "session_ledger.jsonl"
RECAP_QUESTION = "recap the five biggest decisions we made"
RECAP_NGEN = 160


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

class ChatSession:
    """One repository directory, opened under one resolved flag set.

    The repository directory IS the session: closing and re-opening it (a
    ``/restart``, or a fresh process with ``--resume``) restores the memory,
    because the grafts and their manifest are on disk.  Nothing about the
    conversation lives in this object that is needed to answer — only the
    turn counter and the ledger path, both re-derived from disk on resume.
    """

    def __init__(self, repo_dir: Path, resolved: dict[str, Any], *,
                 fake: bool = False) -> None:
        self.repo_dir = Path(repo_dir)
        self.resolved = resolved
        self.fake = bool(fake)
        self.session_id = str(uuid.uuid4())
        self.process_id = self.session_id
        self.turn_idx = 0
        self.repo = None
        self.model_info: dict[str, Any] | None = None
        self.args = None
        self.ledger = self.repo_dir / LEDGER_NAME
        self.opened_at: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Create or resume the repository directory.

        ``GraftRepository.__init__`` calls ``self.load()`` when
        ``manifest.json`` already exists, so resume is *construction on the
        same path* — the same thing ``grm_lt1_worker.execute`` does after it
        copies a checkpoint's repository in.  There is no separate resume
        code path to drift from the fresh one.
        """
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.repo_dir
        flags = self.resolved["flags"]
        if self.fake:
            self.repo, self.model_info = _open_fake_repo(session_dir, flags)
            self.args = _fake_args(flags)
        else:
            from scripts import grm_e2e_session as e2e
            from scripts.grm_c2_cells import args_for
            args = args_for(e2e, session_dir, flags)
            args.session_dir = session_dir
            _, _, repo, info = e2e.load_model_and_repo(args, session_dir)
            self.repo, self.model_info, self.args = repo, info, args
        arena = self.repo.arena
        if not arena.ephemeral:
            raise ValueError("FRAME_NOT_EPHEMERAL: EB1 boat required")
        if int(arena.width) != int(flags["arena_width"]):
            raise ValueError(
                f"PROFILE_GEOMETRY_MISMATCH: arena width {arena.width} "
                f"!= requested {flags['arena_width']}")
        self.turn_idx = _resume_turn_index(self.ledger)
        self.opened_at = time.time()

    def close(self) -> None:
        if self.repo is not None:
            try:
                self.repo.flush_now()
            finally:
                self.repo.close()
                self.repo = None

    def restart(self) -> dict[str, Any]:
        """Close and reopen the repository in-process.

        This is the interactive form of the batteries' restart cells: flush,
        close, drop the object, construct again on the same directory.  The
        node count before and after is the receipt that memory survived.
        """
        before = len(self.repo.arena.grafts)
        previous = self.process_id
        if self.fake:
            _save_fake_codec(self)
        self.close()
        self.process_id = str(uuid.uuid4())
        self.open()
        after = len(self.repo.arena.grafts)
        record = {
            "kind": "restart",
            "previous_process_id": previous,
            "process_id": self.process_id,
            "nodes_before": int(before),
            "nodes_after": int(after),
            "retained": int(before) == int(after),
            "in_process": True,
        }
        if not record["retained"]:
            raise ValueError(f"RESTART_LOST_NODES: {before} -> {after}")
        self._append_ledger({"schema": "grm.chat_event.v1", **record})
        return record

    # -- the turn ----------------------------------------------------------

    def ask(self, user_text: str, *, ngen: int | None = None,
            deposit: bool = True) -> dict[str, Any]:
        """One user turn: answer from memory, then deposit the turn.

        LIVE CONTEXT CONTRACT.  ``_probe_ladder_chat`` reaches
        ``ArenaCache._attempt``, whose prompt is
        ``self.encode(self._format_step_prompt(user_text))`` — that is
        ``harmony_turn(user_text, None)``, THIS TURN ONLY.  Prior turns are
        present only as routed K/V grafts, never as text.  Under the EB1
        ephemeral frame ``live_segs`` is cleared at ``eb1_begin_turn`` and
        nothing refeeds it (``refeed_live_window`` is the persistent-frame
        escape and this surface never calls it).  ``tests/test_grm_chat_
        leak.py`` asserts this on the real serving path.
        """
        from scripts import grm_e2e_session as e2e

        if self.repo is None:
            raise RuntimeError("session is not open")
        arena = self.repo.arena
        flags = self.resolved["flags"]
        self.turn_idx += 1
        turn_idx = self.turn_idx
        started = time.perf_counter()
        # ``before`` is the repository's own state snapshot (a LIST, one
        # tuple per graft). It is what ``runtime._finish_turn_event`` uses
        # to find the nodes this turn appended, so it must be taken before
        # the deposit, and it is also the arena-state hash for the receipt.
        before = self.repo._snapshot_state()
        try:
            from core.grm_three_pass import arena_state_sha256
            arena_before_sha = arena_state_sha256(self.repo)
        except Exception:                                   # pragma: no cover
            arena_before_sha = None

        answer, info = e2e._probe_ladder_chat(
            self.repo, user_text,
            topk=int(flags["topk"]),
            ngen=int(ngen if ngen is not None else flags["ngen"]),
            max_trips=int(flags["max_trips"]),
            defer_memory=True)
        answer = str(answer)
        mounted = [int(i) for i in arena.cur_mounts]
        wall_ms = (time.perf_counter() - started) * 1000.0

        node_id = None
        if deposit:
            # EB1 complete-turn deposit: the question AND the answer go in as
            # one turn node, AFTER the answer exists. Prior art: EB1/LT1 r1
            # frozen replay (GRM, 2026), as grm_lt1_worker.execute does it.
            node_id = int(arena.feed(e2e.harmony_turn(user_text, answer)))
            arena.grafts[node_id]["kind"] = "turn"
            # ...AND THEN THE PRODUCTION TURN FUNNEL. A battery may stop at
            # feed() because it replays a frozen fixture; a product may not.
            # ``runtime._finish_turn_event`` is the funnel every production
            # deposit passes through (grm_e2e_session._probe_finish_deposit
            # calls exactly this): it runs the LSR-P2C width guard, the
            # librarian, and ``repo._mark_mutations(before)`` — which calls
            # ``_mark_dirty(payload=True)`` -> ``_native_sync_node(idx)``,
            # the ONLY thing that assigns ``native_node_id``.
            #
            # Skipping it is what sent GRM-P1's first GPU smoke RED at the
            # first recall: ``_commit_native_mount`` -> ``_native_mount_ids``
            # raised ``RuntimeError: graft 0 has no native_node_id`` because
            # three fed turns had never been published to the native store.
            # LT1 met the same wall on 2026-09-09 and answered it with a
            # HARNESS workaround (grm_lt1_amendment4.install_native_
            # publication monkey-patches _commit_native_mount to publish
            # lazily); that is correct for a fixture replayer and wrong for
            # a product surface, so this calls the real funnel instead.
            self.repo.runtime._finish_turn_event(
                "chat", before, autosave=True)
            if arena.grafts[node_id].get("native_node_id") is None and (
                    getattr(self.repo, "native_store", None) is not None):
                raise RuntimeError(
                    f"NATIVE_PUBLICATION_FAILED: graft {node_id} has no "
                    "native_node_id after the turn funnel")

        event = {"kind": "chat", "user": user_text}
        receipt = e2e._turn_route_receipt(
            self.repo, event, info, answer,
            turn_idx=turn_idx,
            session_id=self.session_id,
            prep_receipt=None,
            arena_before_sha256=arena_before_sha,
            args=self.args)
        row = {
            "schema": "grm.chat_turn.v1",
            "turn": turn_idx,
            "process_id": self.process_id,
            "user": user_text,
            "answer": answer,
            "mounted_ids": mounted,
            "deposited_node_id": node_id,
            "repository_nodes": int(len(arena.grafts)),
            "admission_rule": self.resolved["admission_rule"],
            "profile": self.resolved["profile"],
            "wall_ms": round(wall_ms, 3),
            "route_receipt": receipt,
        }
        self._append_ledger(row)
        return row

    def recap(self) -> dict[str, Any]:
        """The LT1-style five-decision recap, asked over memory.

        Read-only by construction: the recap question is not deposited, so
        asking for a recap never changes what the next turn can recall.
        LT1 scored this 0/5 against its registered decision list — see
        docs/GRM_CHAT.md, "Known residuals".
        """
        return self.ask(RECAP_QUESTION, ngen=RECAP_NGEN, deposit=False)

    # -- introspection -----------------------------------------------------

    def status(self) -> dict[str, Any]:
        arena = self.repo.arena
        grafts = arena.grafts
        mounts = [int(i) for i in arena.cur_mounts]
        try:
            seats = sum(int(grafts[i]["ntok"]) for i in mounts)
        except (KeyError, TypeError, ValueError, IndexError):
            seats = None
        return {
            "repo": str(self.repo_dir),
            "profile": self.resolved["profile"],
            "turns_this_process": self.turn_idx,
            "repository_nodes": len(grafts),
            "seats_used": seats,
            "arena_width": int(arena.width),
            "n_sink": int(getattr(arena, "n_sink", 0) or 0),
            "live_shift": getattr(arena, "live_shift", None),
            "mounted_ids": mounts,
            "mount_seats": int(getattr(arena, "cur_mount_n", 0) or 0),
            "admission_rule": self.resolved["admission_rule"],
            "ephemeral_boat": bool(arena.ephemeral),
            "live_segments": len(getattr(arena, "live_segs", ()) or ()),
            "flags": dict(self.resolved["flags"]),
            "env": dict(self.resolved["env"]),
            "ledger": str(self.ledger),
            "model": self.model_info,
        }

    # -- ledger ------------------------------------------------------------

    def _append_ledger(self, row: dict[str, Any]) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")


def _resume_turn_index(ledger: Path) -> int:
    """Highest turn number already in the ledger (0 for a new session)."""
    if not ledger.is_file():
        return 0
    highest = 0
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("turn"), int):
            highest = max(highest, int(row["turn"]))
    return highest


# --------------------------------------------------------------------------
# Fake (CPU) model — gates only, never a quality measurement
# --------------------------------------------------------------------------

def _fake_args(flags: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        topk=int(flags["topk"]), max_trips=int(flags["max_trips"]),
        ngen=int(flags["ngen"]), arena_width=int(flags["arena_width"]),
        live_turns=int(flags["live_turns"]), max_live=int(flags["max_live"]),
        turn_pipeline=str(flags["turn_pipeline"]), vram_budget_mb=None,
        model_dir=None, native_lib=None, session_dir=None, mode="full")


def _open_fake_repo(session_dir: Path, flags: dict[str, Any]):
    """CPU double: real repository/arena/admission/ladder, stub model.

    Prior art: grm_c7_diagnose.repository + Model (GRM contributors, 2026),
    used verbatim.  Only the numerical model and payload boundaries are
    replaced; routing, admission, mounting, deposit and persistence are the
    production objects.  A fake-model answer is PLUMBING evidence only.
    """
    from contextlib import nullcontext

    import numpy as np

    import core.graft_arena as ga
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e
    from scripts.grm_c7_diagnose import CPUArena, Codec, Model

    ga.tc.no_grad = nullcontext
    ga.tc.cat = lambda values, dim: np.concatenate(values, axis=dim)

    class ChatCPUArena(CPUArena):
        """CPUArena + the mutation-epoch bump its ``deposit`` override drops.

        FAKE-HARNESS DEFECT, FOUND BY THIS SMOKE (GRM-P1, 2026-09-10).
        ``ArenaCache.deposit`` calls ``_bump_cuda_gqa_epoch()`` at every
        ``self.grafts`` mutation; ``grm_c7_diagnose.CPUArena.deposit``
        (scripts/grm_c7_diagnose.py:118) replaces the whole method and does
        not.  ``_route_cand_base`` is epoch-cached, so on the fake harness
        the base list is cached EMPTY at the first route and never rebuilt:
        every pre-restart ``route()`` returns ``[]`` and only a reopen (a
        fresh arena) recovers.  Production is unaffected — it runs
        ``GptOssGQAArenaCache``, whose inherited ``deposit`` bumps.

        ``grm_c7_diagnose.py`` is a SHA-frozen immutable input of the C7-r3
        and LT1 registrations (``artifacts/grm_c7/r3/registration.json``,
        ``artifacts/grm_lt1/registration.json``), so it is NOT edited here.
        The bump is restored in this subclass, which is the same thing
        production does and changes no registered fixture.
        """

        def deposit(self, text, capture_pin=None):
            index = super().deposit(text, capture_pin)
            self._bump_cuda_gqa_epoch()
            return index

    codec = Codec()
    words = session_dir / "codec_words.json"
    if words.is_file():
        # The stub tokenizer's vocabulary is process state, not repository
        # state; a resumed fake session must restore it or the persisted
        # token ids decode to the wrong words.
        codec.words = json.loads(words.read_text())
        codec.ids = {w: i for i, w in enumerate(codec.words)}
    model = Model(codec)
    repo = GraftRepository(
        model, codec.encode, codec.decode, str(session_dir / "repository"),
        autosave=False, native_auto=False, arena_cls=ChatCPUArena,
        arena_width=int(flags["arena_width"]), route_layer=0, ephemeral=True,
        recency_mounts=int(flags["live_turns"]),
        sink_text=e2e.HARMONY_SINK, prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS,
        revision_resolution=bool(flags["sup_resolve"]),
        decisive_admission=bool(flags["adm_decisive"]),
        route_backend="python")
    repo.arena._rs3_seat_explicit = False
    repo._grm_chat_codec_words = words
    return repo, {"backend": "cpu_fake", "model": "grm_c7_diagnose.Model",
                  "evidence_class": "CPU fake model; plumbing only"}


def _save_fake_codec(session: "ChatSession") -> None:
    path = getattr(session.repo, "_grm_chat_codec_words", None)
    if path is not None:
        Path(path).write_text(
            json.dumps(list(session.repo.arena.m.codec.words)) + "\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

BANNER_KEYS = ("GRM_PERSISTENT_BOAT", "GRM_CAPTURE_PIN", "GRM_SEAT_NEAR_LIVE",
               "GRM_LSR_FIXES", "GRM_RT1_RULE", "GRM_DEMAND_NGH",
               "GRM_GQA_CUDA_ROUTE", "GRM_GRAFT_STORAGE_BITS",
               "GRM_ROUTE_QUERY_LEX", "GRM_PROBE_LADDER", "GRM_SUP_RESOLVE",
               "GRM_ADM_DECISIVE", "GRM_ADMISSION_RULE")


def format_resolved(resolved: dict[str, Any]) -> str:
    """The resolved flag set, printed at the start of every session."""
    flags = resolved["flags"]
    lines = [
        f"GRM chat — profile: {resolved['profile']}"
        f"  ({resolved['profile_source']})",
        "  frame            : EB1 ephemeral boat"
        "  (chat log NEVER in context)",
        f"  arena width      : {flags['arena_width']}",
        f"  capture pin      : {flags['capture_pin']}",
        f"  seat near live   : {flags['seat_near_live']}",
        f"  RT1 rule         : {flags['rt1_rule']}",
        f"  admission rule   : {resolved['admission_rule']}",
        f"  topk / ngen      : {flags['topk']} / {flags['ngen']}",
    ]
    for name in resolved.get("named_flags_applied", ()):
        lines.append(f"  named flag       : {name} (pinned)")
    for note in resolved.get("notes", ()):
        lines.append(f"  note             : {note}")
    lines.append("  env:")
    env = resolved["env"]
    for key in BANNER_KEYS:
        if key in env:
            lines.append(f"    {key}={env[key]}")
    return "\n".join(lines)


HELP_TEXT = """commands:
  /status   seats used, width, mounted ids, admission rule, resolved flags
  /recap    the five-decision recap, asked over memory (not deposited)
  /restart  close and reopen the repository in-process (memory must survive)
  /help     this list
  /quit     flush, close, exit
anything else is a chat turn."""


def run_interactive(session: ChatSession, *, stream=sys.stdout) -> int:
    print(format_resolved(session.resolved), file=stream)
    print(f"  repository       : {session.repo_dir}", file=stream)
    print(f"  ledger           : {session.ledger}", file=stream)
    print(f"  nodes in memory  : {len(session.repo.arena.grafts)}",
          file=stream)
    print("\n" + HELP_TEXT + "\n", file=stream)
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("", file=stream)
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/help":
            print(HELP_TEXT, file=stream)
            continue
        if text == "/status":
            print(json.dumps(session.status(), indent=2, default=str),
                  file=stream)
            continue
        if text == "/restart":
            print(json.dumps(session.restart(), indent=2), file=stream)
            continue
        if text == "/recap":
            print(f"grm> {session.recap()['answer']}", file=stream)
            continue
        if text.startswith("/"):
            print(f"unknown command {text!r}; /help for the list", file=stream)
            continue
        print(f"grm> {session.ask(text)['answer']}", file=stream)


def run_transcript(session: ChatSession, transcript: Path, *,
                   stream=sys.stdout) -> dict[str, Any]:
    """Batch mode: one user turn per line; ``/`` commands are honoured."""
    print(format_resolved(session.resolved), file=stream)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for raw in Path(transcript).read_text().splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if text == "/restart":
            events.append({"command": "/restart", **session.restart()})
            print("[restart] memory retained", file=stream)
            continue
        if text == "/status":
            events.append({"command": "/status", **session.status()})
            continue
        if text == "/recap":
            row = session.recap()
        else:
            row = session.ask(text)
        rows.append(row)
        print(f"you> {text}\ngrm> {row['answer']}", file=stream)
    return {"rows": rows, "events": events}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="grm_chat.py",
        description=(
            "Interactive GRM chat. The chat log is NEVER in the model's "
            "context: each turn is served as the question plus routed "
            "grafts only, and is deposited into the repository afterwards "
            "(EB1 ephemeral boat)."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "one switch:\n"
            f"  {PROFILE_ENV}=eb1_c2   select the registered C2 profile\n"
            "                        (width 96, capture pin live, "
            "seat-near-live,\n"
            "                         RT1, margin-first admission).\n"
            "  unset                 today's shipped defaults.\n\n"
            "example:\n"
            "  GRM_PROFILE=eb1_c2 python3 scripts/grm_chat.py "
            "--repo ~/grm_sessions/notes\n"))
    p.add_argument("--repo", type=Path, required=True,
                   help="repository directory for this session "
                        "(created, or resumed if it exists)")
    p.add_argument("--resume", action="store_true",
                   help="require an existing repository (fail if absent)")
    p.add_argument("--transcript", type=Path, default=None,
                   help="batch mode: one user turn per line")
    p.add_argument("--profile", default=None,
                   help=f"override the {PROFILE_ENV} environment switch")
    p.add_argument("--pin-flag", action="append", default=[],
                   metavar="NAME[=VALUE]",
                   help="pin an optional named flag (e.g. "
                        "GRM_ALIAS_FOLD_MERGE); repeatable")
    p.add_argument("--fake-model", action="store_true",
                   help="CPU stub model (gates only; never a quality claim)")
    p.add_argument("--print-flags", action="store_true",
                   help="print the resolved flag set and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolved = resolve_profile(selection=args.profile, pinned=args.pin_flag)
    if args.print_flags:
        print(format_resolved(resolved))
        print(json.dumps(resolved, indent=2, sort_keys=True, default=str))
        return 0

    repo_dir = args.repo.expanduser().resolve()
    if args.resume and not (repo_dir / "repository" / "manifest.json").is_file():
        print(f"--resume: no repository at {repo_dir}", file=sys.stderr)
        return 2

    # The resolved env is pinned into this process BEFORE the arena is
    # built, the same way grm_c2_cells.environment pins a worker's frame.
    os.environ.update(resolved["env"])

    session = ChatSession(repo_dir, resolved, fake=args.fake_model)
    session.open()
    try:
        if args.transcript is not None:
            run_transcript(session, args.transcript)
            return 0
        return run_interactive(session)
    finally:
        if args.fake_model and session.repo is not None:
            _save_fake_codec(session)
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
