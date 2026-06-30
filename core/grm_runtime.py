"""Runtime coordinator for the GRM repository hot path.

This module is deliberately policy-level Python. It gives the repository a
single orchestration boundary for turn execution, extraction, librarian work,
flush, and paging while the C++/CUDA runtime continues to absorb lower-level
storage and cache movement.
"""

from dataclasses import dataclass, field


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

    def _finish_turn_event(self, event, before, extraction=(), *,
                           autosave=False):
        repo = self.repository
        folds_before = len(getattr(repo, "fold_history", ()))
        repo._librarian()
        folds = len(getattr(repo, "fold_history", ())) - folds_before
        repo._mark_mutations(before)
        did_autosave = False
        if autosave and repo.autosave:
            repo.flush_now()
            did_autosave = True
        paged = repo._page()
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

    def chat(self, user_text, ngen=64, max_trips=2):
        repo = self.repository
        before = repo._snapshot_state()
        ans, info = repo.arena.step(user_text, ngen=ngen,
                                    max_trips=max_trips)
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

    def _finish_memory_event(self, before, action, *, force_flush=False):
        repo = self.repository
        did_flush = False
        if force_flush or repo.autosave:
            repo.flush_now()
            did_flush = True
        paged = repo._page()
        self.last_result = RuntimeResult(
            event="memory_command",
            before_nodes=len(before),
            after_nodes=len(repo.arena.grafts),
            new_nodes=self._new_nodes(before),
            autosaved=did_flush,
            paged=int(paged or 0),
            action=action,
        )

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
            self._finish_memory_event(
                before, "remember",
                force_flush=bool(plan.get("flush_immediately")))
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
        if action == "ignore":
            repo._append_wal("DO_NOT_REMEMBER", text=text)
            self._finish_memory_event(before, "ignore")
            return {"action": "ignore"}
        if action == "flush":
            self._finish_memory_event(before, "flush", force_flush=True)
            return {"action": "flush"}
        raise ValueError(f"unknown memory command: {text!r}")

    def _finish_review_event(self, before, action):
        repo = self.repository
        did_autosave = False
        if repo.autosave:
            repo.flush_now()
            did_autosave = True
        paged = repo._page()
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

    def _finish_extraction_event(self, before, action, results):
        repo = self.repository
        did_autosave = False
        if repo.autosave:
            repo.flush_now()
            did_autosave = True
        paged = repo._page()
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
