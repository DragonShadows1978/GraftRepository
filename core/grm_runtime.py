"""Runtime coordinator for the GRM repository hot path.

This module is deliberately policy-level Python. It gives the repository a
single orchestration boundary for turn execution, extraction, librarian work,
flush, and paging while the C++/CUDA runtime continues to absorb lower-level
storage and cache movement.
"""

from dataclasses import dataclass, field

from core.grm_three_pass import arena_state_sha256, build_route_receipt

# LSR-P2B: the production serve returns this record under this ``info`` key
# when the repository has no ledger location configured.  ``route_receipt`` is
# already the arena's optional route-profiling blob; a new fact takes a new
# name rather than overloading an existing one.
ROUTE_RECEIPT_INFO_KEY = "route_receipt_record"
ROUTE_RECEIPT_SINK_INFO_KEY = "route_receipt_sink"


@dataclass(frozen=True)
class RuntimeResult:
    event: str
    before_nodes: int
    after_nodes: int
    new_nodes: tuple = field(default_factory=tuple)
    extraction: tuple = field(default_factory=tuple)
    folds: int = 0
    autosaved: bool = False
    paged: int = 0
    action: str = ""


class GRMRuntime:
    """Own the GRM operation sequence above model-specific cache mechanics."""

    def __init__(self, repository):
        self.repository = repository
        self.last_result = None

    def _new_nodes(self, before):
        repo = self.repository
        return tuple(range(len(before), len(repo.arena.grafts)))

    def _runtime_event_plan(self, event, action="", *, autosave_enabled=False,
                            force_flush=False, read_only=False):
        repo = self.repository
        native = getattr(repo, "native_store", None)
        if native is not None and hasattr(native, "plan_runtime_event"):
            try:
                return native.plan_runtime_event(
                    event=event, action=action,
                    autosave_enabled=bool(autosave_enabled),
                    force_flush=bool(force_flush),
                    read_only=bool(read_only))
            except RuntimeError:
                pass
        actual_read_only = bool(read_only) or (
            event == "memory_command" and action in (
                "show_memory", "why_memory"))
        flush = False
        page = not actual_read_only
        if not actual_read_only:
            flush = (bool(force_flush) or
                     (event == "memory_command" and action == "flush") or
                     bool(autosave_enabled))
        return {"flush": flush, "page": page, "read_only": actual_read_only}

    def _guard_deposit_width(self, before):
        """LSR-P2C Part 1: width-guard every node this event deposited.

        ``before`` is the ``_snapshot_state()`` list the event opened with, so
        ``len(before)`` is the graft count before any deposit.  Running the
        guard HERE (before ``_librarian()`` and before ``_mark_mutations``)
        means the librarian, the extractor, the WAL and the pager all see the
        already-split shape — a node the arena cannot mount never reaches
        them.

        The repository object is duck-typed on purpose: the three-pass and
        diagnostic harnesses build runtimes over stub repositories that have
        no width guard, and those must keep working unchanged.
        """
        guard = getattr(self.repository, "_guard_deposit_range", None)
        if not callable(guard):
            return ()
        return guard(len(before))

    def _finish_turn_event(self, event, before, extraction=(), *,
                           autosave=False):
        repo = self.repository
        # LSR-P2C Part 1: the LAST funnel every turn-deposit event passes
        # through, including the DEFERRED-TURN path whose deposit happens
        # outside chat()/add_turn() (grm_e2e_session._commit_turn_mutations ->
        # arena.deposit_deferred_turn, then runtime._finish_turn_event).
        # Guarding here means the librarian below never folds an unmountable
        # node into a digest.
        self._guard_deposit_width(before)
        folds_before = len(getattr(repo, "fold_history", ()))
        repo._librarian()
        folds = len(getattr(repo, "fold_history", ())) - folds_before
        repo._mark_mutations(before)
        plan = self._runtime_event_plan(
            event, autosave_enabled=bool(autosave and repo.autosave))
        did_autosave = False
        if plan.get("flush"):
            repo.flush_now()
            did_autosave = True
        paged = repo._page() if plan.get("page", True) else 0
        result = RuntimeResult(
            event=event,
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            extraction=tuple(extraction or ()),
            folds=folds,
            autosaved=did_autosave,
            paged=int(paged or 0),
        )
        self.last_result = result
        return result

    def _route_receipt_turn_id(self):
        """Monotonic per-process turn ordinal for the production serve."""
        ordinal = int(getattr(self, "_route_receipt_turns", 0))
        self._route_receipt_turns = ordinal + 1
        return str(ordinal)

    def _persist_route_receipt(self, record, info):
        """Write to the repository's ledger location, else return in info.

        LSR-P2B principle: the receipt is produced unconditionally.  Where it
        lands is a deployment question, and the answer is always stated in
        ``info[ROUTE_RECEIPT_SINK_INFO_KEY]`` so a reader never has to guess
        whether a missing file means "not configured" or "not written".
        """
        repo = self.repository
        writer = getattr(repo, "write_route_receipt", None)
        if callable(writer):
            try:
                path = writer(record)
            except Exception as exc:
                info[ROUTE_RECEIPT_INFO_KEY] = record
                info[ROUTE_RECEIPT_SINK_INFO_KEY] = {
                    "sink": "info", "reason": "ledger_write_failed",
                    "error": repr(exc)}
                return info
            if path is not None:
                info[ROUTE_RECEIPT_SINK_INFO_KEY] = {
                    "sink": "repository_ledger", "path": str(path)}
                return info
            # Writer present but no location configured: fall through to info.
        info[ROUTE_RECEIPT_INFO_KEY] = record
        info[ROUTE_RECEIPT_SINK_INFO_KEY] = {
            "sink": "info", "reason": "no_ledger_location_configured"}
        return info

    def chat(self, user_text, ngen=64, max_trips=2):
        """The production serve.

        GRM-EB1 FRAME.  This path serves under whatever frame the repository's
        arena was built with, and since ``ArenaCache`` now defaults to the
        SPEC frame (ephemeral boat), every runtime built the ordinary way
        serves without a persistent live window: no prior turn's tokens sit in
        the model's context, and every recall of an earlier fact is a routed
        graft.  ``step()`` attaches the frame receipt (``frame_ephemeral``,
        ``frame_escape_active``, ``recency_mounted_ids``, ``recency_seats``,
        ``live_segments_after_turn``) to ``info``, and ``build_route_receipt``
        persists it through the ``frame_``/``recency_``/``live_segments_``
        prefixes, so the frame a served turn ran under is on the receipt.
        """
        repo = self.repository
        before = repo._snapshot_state()
        # LSR-P2B: arena state as the serve saw it, captured before step().
        try:
            arena_before_sha256 = arena_state_sha256(repo)
        except Exception:
            arena_before_sha256 = None
        route_limit = max(1, (int(max_trips) + 1) * int(
            getattr(repo.arena, "topk", 1) or 1))
        ans, info = repo.arena.step(user_text, ngen=ngen,
                                    max_trips=max_trips)
        record = build_route_receipt(
            session_id=str(getattr(repo, "path", "") or "repository"),
            turn_id=self._route_receipt_turn_id(),
            info=info,
            arena=repo.arena,
            request_text=user_text,
            output_text=ans,
            arena_before_sha256=arena_before_sha256,
            serving_path="core.grm_runtime.GRMRuntime.chat",
            route_limit=route_limit,
            turn_kind="chat",
        )
        info = self._persist_route_receipt(record, info)
        # LSR-P2C Part 1: the turn graft arena.step() just deposited is
        # width-guarded BEFORE extraction reads it.  _finish_turn_event runs
        # the same guard again for the deferred-turn path (whose deposit
        # happens outside chat()); the second pass is a no-op integer scan
        # because children are budget-fitting by construction.
        self._guard_deposit_width(before)
        extracted = repo._extract_from_new_turns(
            before, context={"event": "chat", "user_text": user_text,
                             "assistant_text": ans})
        if extracted:
            info["extraction"] = extracted
        self._finish_turn_event(
            "chat", before, extraction=extracted, autosave=True)
        return ans, info

    def add_turn(self, user, assistant):
        repo = self.repository
        before = repo._snapshot_state()
        repo.arena.feed(f"User: {user}\nAssistant: {assistant}\n")
        repo._set_new_node_provenance(before, "exchange_span")
        # LSR-P2C Part 1: a fed turn longer than the arena can seat is split
        # here, before extraction or the librarian sees it.
        self._guard_deposit_width(before)
        extracted = repo._extract_from_new_turns(
            before, context={"event": "add_turn", "user_text": user,
                             "assistant_text": assistant})
        return self._finish_turn_event(
            "add_turn", before, extraction=extracted, autosave=False)

    def idle(self, max_jobs=1):
        repo = self.repository
        before = repo._snapshot_state()
        done = 0
        while done < max_jobs and repo._fold_once():
            done += 1
        if done:
            repo._mark_mutations(before)
            did_autosave = False
            if repo.autosave:
                repo.flush_now()
                did_autosave = True
            paged = repo._page()
        else:
            did_autosave = False
            paged = 0
        self.last_result = RuntimeResult(
            event="idle",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            folds=done,
            autosaved=did_autosave,
            paged=int(paged or 0),
        )
        return done

    def _finish_memory_event(self, before, action, *, force_flush=False,
                             read_only=False):
        repo = self.repository
        plan = self._runtime_event_plan(
            "memory_command", action, autosave_enabled=repo.autosave,
            force_flush=force_flush, read_only=read_only)
        did_flush = False
        if plan.get("flush"):
            repo.flush_now()
            did_flush = True
        paged = repo._page() if plan.get("page", True) else 0
        self.last_result = RuntimeResult(
            event="memory_command",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            autosaved=did_flush,
            paged=int(paged or 0),
            action=action,
        )

    @staticmethod
    def _remember_force_flush(repo, plan):
        if hasattr(repo, "_native_remember_flush_plan"):
            native_plan = repo._native_remember_flush_plan(plan)
            if native_plan is not None:
                return bool(native_plan.get("force_flush"))
        if bool(plan.get("flush_immediately")):
            return True
        if repo.durability_mode != "project_safe":
            return False
        durability = plan.get("durability")
        scope = plan.get("scope")
        return durability in ("project", "permanent") or scope == "project"

    @staticmethod
    def _metadata_updates(repo, plan):
        updates = dict(plan.get("metadata", {}) or {})
        if hasattr(repo, "_native_metadata_update_plan"):
            native_plan = repo._native_metadata_update_plan(plan)
            if native_plan is not None:
                updates.update(dict(native_plan.get("metadata", {}) or {}))
                return updates
        key = plan.get("metadata_key")
        if key:
            value = plan.get("metadata_value")
            if value == "true":
                value = True
            elif value == "false":
                value = False
            updates[str(key)] = value
        return updates

    def apply_memory_command(self, text):
        repo = self.repository
        before = repo._snapshot_state()
        plan = repo._parse_memory_command(text)
        action = plan.get("action")
        if action == "remember":
            opts = {k: plan[k] for k in ("durability", "mutability",
                                         "scope", "kind")
                    if plan.get(k)}
            idx = repo.remember(plan.get("body", ""), **opts)
            force_flush = self._remember_force_flush(repo, plan)
            self._finish_memory_event(
                before, "remember",
                force_flush=force_flush)
            return {"action": "remember", "node_id": idx}
        if action == "forget":
            count = repo.forget(plan.get("query", ""))
            self._finish_memory_event(before, "forget")
            return {"action": "forget", "count": count}
        if action == "correct":
            idx = repo.correct_memory(
                plan.get("query", ""), plan.get("replacement", ""))
            self._finish_memory_event(before, "correct")
            return {"action": "correct", "node_id": idx}
        if action == "review":
            repo.review_candidate(
                plan.get("body", ""),
                action="review_candidate",
                reason=plan.get("reason",
                                "correction missing => separator"))
            self._finish_memory_event(before, "review")
            return {"action": "review", "count": len(repo.review_buffer)}
        if action == "approve_review":
            review_id = plan.get("review_id")
            if review_id is None:
                raise ValueError("approve_review command requires review_id")
            idx = repo._approve_review_direct(int(review_id))
            self._finish_memory_event(before, "approve_review")
            return {"action": "approve_review", "review_id": int(review_id),
                    "node_id": idx}
        if action == "reject_review":
            review_id = plan.get("review_id")
            if review_id is None:
                raise ValueError("reject_review command requires review_id")
            item = repo._reject_review_direct(
                int(review_id), reason=plan.get("reason", ""))
            self._finish_memory_event(before, "reject_review")
            return {"action": "reject_review", "review_id": int(review_id),
                    "item": item}
        if action == "edit_review":
            review_id = plan.get("review_id")
            if review_id is None:
                raise ValueError("edit_review command requires review_id")
            item = repo._edit_review_direct(
                int(review_id), text=plan.get("body", ""),
                reason=plan.get("reason", "memory command edit"))
            self._finish_memory_event(before, "edit_review")
            return {"action": "edit_review", "review_id": int(review_id),
                    "item": item}
        if action == "change_review_scope":
            review_id = plan.get("review_id")
            if review_id is None:
                raise ValueError(
                    "change_review_scope command requires review_id")
            item = repo._edit_review_direct(
                int(review_id), proposed_scope=plan.get("scope"),
                proposed_durability=plan.get("durability") or None,
                proposed_mutability=plan.get("mutability") or None,
                reason="scope changed")
            self._finish_memory_event(before, "change_review_scope")
            return {"action": "change_review_scope",
                    "review_id": int(review_id), "item": item}
        if action == "ignore":
            repo._append_wal("DO_NOT_REMEMBER", text=text)
            self._finish_memory_event(before, "ignore")
            return {"action": "ignore"}
        if action == "flush":
            self._finish_memory_event(before, "flush", force_flush=True)
            return {"action": "flush"}
        if action == "cull_graft":
            node_id = plan.get("node_id")
            if node_id is None:
                raise ValueError("cull_graft command requires node_id")
            kwargs = {}
            if plan.get("max_tokens") is not None:
                kwargs["max_tokens"] = int(plan["max_tokens"])
            boundary = plan.get("boundary")
            if boundary:
                spans = repo.plan_cull_sections(
                    int(node_id), boundary=boundary, **kwargs)
                kwargs.pop("max_tokens", None)
                kwargs["spans"] = spans
            out = repo._cull_graft_direct(int(node_id), **kwargs)
            self._finish_memory_event(before, "cull_graft")
            return out
        if action == "split_oversized":
            # LSR-P2C Part 3: the one-time legacy repair sweep.
            kwargs = {}
            if plan.get("boundary"):
                kwargs["boundary"] = str(plan["boundary"])
            if plan.get("budget") is not None:
                kwargs["budget"] = int(plan["budget"])
            out = repo.split_oversized(**kwargs)
            self._finish_memory_event(before, "split_oversized")
            return out
        if action == "select_graft_span":
            node_id = plan.get("node_id")
            start = plan.get("span_start")
            end = plan.get("span_end")
            if node_id is None or start is None or end is None:
                raise ValueError(
                    "select_graft_span command requires node_id/span bounds")
            out = repo._select_graft_span_direct(
                int(node_id), int(start), int(end), label=plan.get("body", ""))
            self._finish_memory_event(before, "select_graft_span")
            return out
        if action == "update_memory_metadata":
            updates = self._metadata_updates(repo, plan)
            out = repo.update_memory_metadata(plan.get("query", ""), updates)
            label = plan.get("command", "update_memory_metadata")
            self._finish_memory_event(before, label)
            out["action"] = label
            return out
        if action == "show_memory":
            rows = repo.show_memory_about(plan.get("query", ""))
            self._finish_memory_event(before, "show_memory", read_only=True)
            return {"action": "show_memory", "rows": rows}
        if action == "why_memory":
            rows = repo.why_remember(plan.get("query", ""))
            self._finish_memory_event(before, "why_memory", read_only=True)
            return {"action": "why_memory", "rows": rows}
        if action == "set_durability_mode":
            out = repo.set_durability_mode(plan.get("durability_mode", ""))
            self._finish_memory_event(before, "set_durability_mode")
            out["action"] = "set_durability_mode"
            return out
        raise ValueError(f"unknown memory command: {text!r}")

    def _finish_review_event(self, before, action):
        repo = self.repository
        plan = self._runtime_event_plan(
            "review", action, autosave_enabled=repo.autosave)
        did_autosave = False
        if plan.get("flush"):
            repo.flush_now()
            did_autosave = True
        paged = repo._page() if plan.get("page", True) else 0
        self.last_result = RuntimeResult(
            event="review",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            autosaved=did_autosave,
            paged=int(paged or 0),
            action=action,
        )

    def edit_review(self, review_id, **kwargs):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._edit_review_direct(review_id, **kwargs)
        self._finish_review_event(before, "edit_review")
        return out

    def reject_review(self, review_id, reason=""):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._reject_review_direct(review_id, reason=reason)
        self._finish_review_event(before, "reject_review")
        return out

    def approve_review(self, review_id):
        repo = self.repository
        before = repo._snapshot_state()
        idx = repo._approve_review_direct(review_id)
        self._finish_review_event(before, "approve_review")
        return idx

    def _finish_cull_event(self, before, action):
        repo = self.repository
        plan = self._runtime_event_plan(
            "cull", action, autosave_enabled=repo.autosave)
        did_autosave = False
        if plan.get("flush"):
            repo.flush_now()
            did_autosave = True
        paged = repo._page() if plan.get("page", True) else 0
        self.last_result = RuntimeResult(
            event="cull",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            autosaved=did_autosave,
            paged=int(paged or 0),
            action=action,
        )

    def cull_graft(self, idx, **kwargs):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._cull_graft_direct(idx, **kwargs)
        self._finish_cull_event(before, out.get("action", "cull_graft"))
        return out

    def select_graft_span(self, idx, start, end, **kwargs):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._select_graft_span_direct(idx, start, end, **kwargs)
        self._finish_cull_event(before, "select_graft_span")
        return out

    def _finish_extraction_event(self, before, action, results):
        repo = self.repository
        plan = self._runtime_event_plan(
            "extraction", action, autosave_enabled=repo.autosave)
        did_autosave = False
        if plan.get("flush"):
            repo.flush_now()
            did_autosave = True
        paged = repo._page() if plan.get("page", True) else 0
        self.last_result = RuntimeResult(
            event="extraction",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            extraction=tuple(results or ()),
            autosaved=did_autosave,
            paged=int(paged or 0),
            action=action,
        )

    def apply_extraction_candidate(self, candidate, **kwargs):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._apply_extraction_candidate_direct(candidate, **kwargs)
        self._finish_extraction_event(before, out.get("action", ""), (out,))
        return out

    def apply_extraction_candidates(self, candidates, **kwargs):
        repo = self.repository
        before = repo._snapshot_state()
        out = repo._apply_extraction_candidates_direct(candidates, **kwargs)
        action = "apply_extraction_candidates"
        if len(out) == 1:
            action = out[0].get("action", action)
        self._finish_extraction_event(before, action, out)
        return out
