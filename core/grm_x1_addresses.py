"""Opt-in GRM-X1 address sidecar; importing it installs no hooks.

Prior art: Codd (1970), relational identity; Denning (1968), working sets
and page absence (https://denninginstitute.com/pjd/PUBS/WSModel_1968.pdf).
Reed (1978), multiversion indirection, unverified — lead to check:
"Naming and Synchronization in a Decentralized Computer System".
Borrowed: logical keys, monotone versions, physical-page indirection.
New here: the GRM sidecar/API and its registered experiment, not these ideas.
Only fixture-owned metadata enters the index; no answers enter the prompt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import unicodedata

ENV_NAME = "GRM_X1_ADDRESSES"
ABSTENTION = "I cannot answer from the mounted memory: PAGE_FAULT."


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return str(env.get(ENV_NAME, "")).strip().casefold() in {"1", "true", "yes", "on"}


def normalize(value):
    # Prior art: Unicode UAX #15 (normalization; unverified — lead to check).
    # Borrow NFKC/casefold; no fuzzy matching or learned alias rule.
    if not isinstance(value, str) or not value.strip():
        raise ValueError("address components must be nonempty strings")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True, order=True)
class Address:
    entity: str
    relation: str

    def __post_init__(self):
        object.__setattr__(self, "entity", normalize(self.entity))
        object.__setattr__(self, "relation", normalize(self.relation))


@dataclass(frozen=True)
class Version:
    address: Address
    version: int
    page_ids: tuple[int, ...]
    source_sha256: str


class AddressIndex:
    """Single-writer immutable-version index, scoped to ONE repository.

    Current means highest explicitly published revision, not latest arrival,
    lexical rank, or a timestamp. Out-of-order older writes are rejected.
    No concurrency/persistence atomicity claim across repository and sidecar.
    """
    def __init__(self, repository_id: str):
        self.repository_id = normalize(repository_id)
        self._versions = {}

    def publish(self, address, version, page_ids, source_sha256):
        if not isinstance(address, Address):
            raise TypeError("an Address is required")
        if type(version) is not int or version < 1:
            raise ValueError("version must be a positive integer")
        pages = tuple(page_ids)
        if not pages or any(type(p) is not int or p < 0 for p in pages) or len(set(pages)) != len(pages):
            raise ValueError("page IDs must be unique nonnegative integers, nonempty")
        if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
            raise ValueError("source_sha256 must be a lowercase SHA-256")
        row = Version(address, version, pages, source_sha256)
        history = self._versions.get(address, ())
        if history and version == history[-1].version and row == history[-1]:
            return row  # Idempotent same-version deposit receipt.
        if history and version <= history[-1].version:
            raise ValueError("immutable version conflict or stale publication")
        self._versions[address] = (*history, row)
        return row

    def resolve(self, address):
        history = self._versions.get(address, ())
        return history[-1] if history else None

    def dump(self):
        return {"schema": "grm.x1.address-index.v1", "repository_id": self.repository_id,
                "versions": [asdict(v) for key in sorted(self._versions) for v in self._versions[key]]}

    def save(self, path):
        # Create-only snapshot; publication/atomic replacement is the caller's job.
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.dump(), stream, sort_keys=True, indent=2)
            stream.write("\n")

    @classmethod
    def restore(cls, payload, *, repository_id):
        if payload["schema"] != "grm.x1.address-index.v1":
            raise ValueError("unknown address schema")
        out = cls(repository_id)
        if out.repository_id != payload["repository_id"]:
            raise ValueError("address index belongs to a different repository")
        for row in payload["versions"]:
            out.publish(Address(**row["address"]), row["version"], row["page_ids"], row["source_sha256"])
        return out


class PageFault(RuntimeError):
    pass


class AddressedRecall:
    """Additive serve/deposit adapter, DEFAULT OFF.

    fallback(question, **kwargs) is the unchanged router service.
    mount_read(question, pages, before_forward, **kwargs) must invoke
    before_forward(ACTUAL mounted IDs) before its first model forward.
    C checks both availability and physical mounts before generation.
    The supplied arena adapter below owns that check for real native reads.
    """
    def __init__(self, index, fallback, mount_read, available_pages, *, environ=None):
        self.index, self.fallback, self.mount_read = index, fallback, mount_read
        self.available_pages = available_pages
        self.environ = environ

    def record_deposit(self, address, version, page_ids, source_sha256):
        if not enabled(self.environ):
            return None
        return self.index.publish(address, version, page_ids, source_sha256)

    def serve(self, question, *, address=None, point_lookup=False, page_fault=False, **kwargs):
        if not enabled(self.environ):
            # OFF does not touch index, enumerate pages, wrap callbacks, copy the
            # return, or change even a single fallback argument.
            return self.fallback(question, **kwargs)
        if address is None and not point_lookup:
            return self.fallback(question, **kwargs)  # Topical discovery unchanged.
        row = self.index.resolve(address) if address is not None else None
        missing = [] if row is None else sorted(set(row.page_ids) - set(self.available_pages()))
        reason = "unresolved_address" if row is None else ("pages_absent" if missing else None)
        info = {"x1_address": asdict(address) if address is not None else None,
                "x1_version": row.version if row else None,
                "x1_requested_pages": list(row.page_ids) if row else [],
                "x1_missing_pages": missing, "x1_page_fault": False,
                "x1_structural_demand_fired": bool(reason), "demand_fired": False}

        def fault(why):
            return ABSTENTION, {**info, "x1_page_fault": True,
                "x1_structural_demand_fired": True, "x1_fault_reason": why,
                "x1_generation_calls": 0, "abstain": True}

        if reason:
            if page_fault:
                return fault(reason)
            answer, fallback_info = self.fallback(question, **kwargs)
            return answer, {**fallback_info, **info, "x1_fallback_reason": reason,
                            "demand_fired": bool(fallback_info.get("demand_fired", False))}
        observed = []

        def before_forward(mounted):
            observed[:] = list(mounted)
            if page_fault and tuple(mounted) != row.page_ids:
                raise PageFault("current_version_not_mounted")

        try:
            answer, result = self.mount_read(question, row.page_ids, before_forward, **kwargs)
        except PageFault as error:
            if not page_fault:
                answer, fallback_info = self.fallback(question, **kwargs)
                return answer, {**fallback_info, **info, "x1_fallback_reason": str(error),
                                "x1_structural_demand_fired": True,
                                "demand_fired": bool(fallback_info.get("demand_fired", False))}
            return fault(str(error))
        return answer, {**result, **info, "x1_actual_pages": observed,
                        "x1_address_coverage": tuple(observed) == row.page_ids,
                        "x1_generation_calls": 1}


def arena_mount_read(arena, question, pages, before_forward, *, ngen=24, max_trips=2):
    """Reuse EB1/RS3 and native _attempt (house 2026), with a pre-forward check.

    This adapter is single-threaded and scoped to a private experimental arena.
    A finally block restores the forward method even if mounting faults.
    The call to _attempt handles real L2 resolution and native page mounting;
    the guard reads cur_mounts AFTER that commit and BEFORE prompt generation.
    """
    if sum(int(arena.grafts[i]["ntok"]) for i in pages) > int(arena.width):
        raise PageFault("unseatable_current_version")
    arena.eb1_begin_turn()
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    own_forward = arena.__dict__.get("_forward")
    had_own_forward = "_forward" in arena.__dict__
    original = arena._forward
    checked = False

    def guarded(*args, **kwargs):
        nonlocal checked
        if not checked:
            before_forward(tuple(arena.cur_mounts))
            checked = True
        return original(*args, **kwargs)

    arena._forward = guarded
    try:
        answer, info = arena._attempt(question, list(pages), ngen, False,
                                      arena.stop_sequences or (), defer_memory=True)
        if not checked:
            raise RuntimeError("native mount adapter made no checked forward")
        return answer, {**info, **arena._eb1_frame_info()}
    finally:
        if had_own_forward:
            arena._forward = own_forward
        else:
            del arena._forward


def resolve_alias(question, aliases):
    # Prior art: dictionary/entity lookup and exact token-span matching;
    # Codd (1970) for lookup; Unicode UAX15 for normalization (unverified lead).
    # New: only the frozen X1 alias vocabulary. Ambiguous/multi-address -> None.
    # No fuzzy, learned, or unrestricted natural-language resolution claim.
    def tokens(value):
        return tuple(re.findall(r"\w+", normalize(value)))

    query = tokens(question)
    def matches(table):
        found = []
        for canonical, variants in table.items():
            for variant in variants:
                span = tokens(variant)
                if any(query[i:i + len(span)] == span for i in range(len(query) - len(span) + 1)):
                    found.append(canonical)
                    break
        return found
    entities, relations = matches(aliases["entities"]), matches(aliases["relations"])
    return Address(entities[0], relations[0]) if len(entities) == len(relations) == 1 else None
