"""Experimental relational witnesses; no GPU imports and no import-time hooks.

Prior art (verified primary sources, 2026-09-08): Fader, Soderland & Etzioni
(2011), ReVerb, https://aclanthology.org/D11-1142/ : relation triples and
syntactic constraints, NOT their extractor implementation. Buneman, Khanna
& Tan (2001), Why and Where, https://www.pure.ed.ac.uk/ws/files/16509989/Why_and_Where_A_Characterization_of_Data_Provenance.pdf :
source-location provenance. Green, Karvounarakis & Tannen (2007), Provenance
Semirings, https://web.cs.ucdavis.edu/~green/papers/pods07.pdf : conjunction
of evidence for joins, NOT a semiring implementation. House SC1.1 (2026)
provides normalize_glyphs. Ours: a limited hand grammar, versioned spans
beside native K/V, complete-clause rejection, and explicit join receipts.

Opt-in integration: install(arena) BEFORE deposits, then gate_answer AFTER
generation and BEFORE committing answer memory. Existing core is untouched.
Setting the flag alone does not install hooks on unrelated serving paths.
Unsupported prose fails closed; this is not a general entailment engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import re
from types import MethodType
from typing import Mapping

from core.grm_text_norm import normalize_glyphs

FLAG = "GRM_X2_WITNESSES"
META = "grm_x2_witnesses"
ABSTENTION = "I do not have a supported answer in the mounted sources."
SCHEMA = "grm.x2.witnesses.v1"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return str((os.environ if environ is None else environ).get(FLAG, "")).strip().casefold() in {"1", "true", "yes", "on"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(text: str) -> str:
    return " ".join(normalize_glyphs(text).replace("’", "'").casefold().strip(" .,:;?!\"“”").split())


# Prior art: ReVerb (Fader et al., 2011), restricted phrase extraction;
# this explicit alias table and grammar are ours, with no learned weights.
ALIASES = {"colour": "color", "hue": "color", "proprietor": "owner", "site": "location"}
RELATIONS = ("owner", "location", "color", "colour", "hue", "manager", "supplier", "capital",
             "token", "dock", "key", "code", "pin", "bridge", "seal", "port", "tone", "mark", "ledger", "pass", "proprietor", "site")
R = "(?:" + "|".join(RELATIONS) + ")"
E = r"[A-Za-z][A-Za-z0-9'’‑‐ -]*?"
NEG = r"(?P<neg>not\s+)?"
COP = r"(?:is|equals|remains|reads|has\s+been\s+set\s+to|is\s+set\s+to)"
_RULES = [
    (rf"(?:the\s+)?(?:current\s+)?(?P<relation>{R})\s+(?:of|for|assigned\s+to)\s+(?P<entity>{E})\s+{COP}\s+{NEG}(?P<value>.+)", None),
    (rf"(?:the\s+)?(?:current\s+)?(?P<entity>{E})(?:'s|’s)?\s+(?P<relation>{R})(?:\s+value)?\s+{COP}\s+{NEG}(?P<value>.+)", None),
    (rf"for\s+(?P<entity>{E}),\s*(?:the\s+)?(?P<relation>{R})(?:\s+value)?\s+{COP}\s+{NEG}(?P<value>.+)", None),
    (rf"(?P<entity>{E})\s+uses\s+(?P<value>.+?)\s+as\s+its\s+(?P<relation>{R})", None),
    (rf"(?P<value>.+?)\s+is\s+{NEG}the\s+(?P<relation>{R})\s+of\s+(?P<entity>{E})", None),
    (rf"(?P<value>.+?)\s+(?P<verb>owns|manages|supplies)\s+(?P<entity>{E})", "active"),
    (rf"(?P<value>.+?)\s+does\s+(?P<neg>not)\s+(?P<verb>own|manage|supply)\s+(?P<entity>{E})", "active"),
    (rf"(?P<entity>{E})\s+is\s+{NEG}(?P<verb>owned\s+by|managed\s+by|supplied\s+by|located\s+in|based\s+in|colored|coloured)\s+(?P<value>.+)", "passive"),
]
RULES = [(re.compile(p, re.I), kind) for p, kind in _RULES]
VERBS = {"owns": "owner", "own": "owner", "manages": "manager", "manage": "manager",
         "supplies": "supplier", "supply": "supplier", "owned by": "owner", "managed by": "manager",
         "supplied by": "supplier", "located in": "location", "based in": "location", "colored": "color", "coloured": "color"}
# No substring extraction from conditional/reported/quoted claims; source
# clauses may be ignored but every answer clause must be fully understood.
FORBIDDEN = re.compile(r"\b(?:not|never|no|neither|nor|unless|if|maybe|perhaps|probably|allegedly|possibly|either|or|but|and|because|although|says|said|claims|claimed|denies|denied|is|are|was|were|the|current|known|unknown|available|unavailable)\b", re.I)


@dataclass(frozen=True)
class Binding:
    entity: str
    relation: str
    value: str
    negated: bool = False


def atom(text: str) -> bool:
    text = canonical(text)
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9' -]*", text)) and len(text.split()) <= 5 and not FORBIDDEN.search(text)


def parse_clause(text: str):
    """Full match only. Return binding and original group offsets, or None."""
    for regex, kind in RULES:
        m = regex.fullmatch(text)
        if not m:
            continue
        d = m.groupdict()
        entity, value = canonical(d["entity"]), canonical(d["value"])
        if entity.endswith("'s"):
            entity = entity[:-2]
        if not atom(entity) or not atom(value):
            continue
        relation = VERBS[canonical(d["verb"])] if kind else canonical(d["relation"])
        relation = ALIASES.get(relation, relation)
        spans = {key: m.span(key) for key in ("entity", "value")}
        spans["relation"] = m.span("verb" if kind else "relation")
        return Binding(entity, relation, value, bool(d.get("neg"))), spans
    return None


def clauses(text: str):
    """Yield exact raw clause slices. Conjunction requires separate bindings.

    Prior art: Thompson (1968), regex search (unverified — lead to check:
    Thompson 1968 regular expression search). Ours: conservative delimiters
    and a small allowlist of dialogue prefixes, no substring rescue.
    """
    for match in re.finditer(r"[^.!?;\n]+", text):
        raw, start = match.group(), match.start()
        # Do not lift a conjunct out of conditional/reported scope.
        if re.search(r"\b(?:if|unless|maybe|perhaps|probably|allegedly|possibly|says|said|claims|claimed|denies|denied)\b", raw, re.I):
            yield raw.strip(), start + len(raw) - len(raw.lstrip())
            continue
        offset = 0
        for piece in re.split(r"\s+and\s+", raw, flags=re.I):
            at = raw.find(piece, offset)
            offset = at + len(piece)
            left = len(piece) - len(piece.lstrip())
            body = piece.strip()
            prefix = re.match(r"(?:(?:User|Assistant|Recorded|Confirmed|Updated):\s*)+", body, re.I)
            if prefix:
                left += prefix.end()
                body = body[prefix.end():]
            if body:
                yield body, start + at + left


def extract(text: str, source_id: str, source_version: str | None = None) -> dict:
    """Deposit-time extraction, with offsets in the ORIGINAL source bytes' text.

    Offsets are Unicode character offsets, not UTF-8 byte offsets. An entire
    binding comes from one contiguous clause in one source version.
    """
    if not source_id:
        raise ValueError("nonempty source_id required")
    version = source_version or sha(text)
    if version != sha(text):
        raise ValueError("source_version must equal the SHA-256 of the deposited text")
    witnesses, skipped = [], []
    for clause, start in clauses(text):
        parsed = parse_clause(clause)
        if parsed is None:
            skipped.append({"start": start, "end": start + len(clause), "text": clause})
            continue
        binding, spans = parsed
        row = {"binding": asdict(binding), "source_id": source_id, "source_version": version,
               "span": [start, start + len(clause)], "text": clause,
               "fields": {k: {"span": [start + a, start + b], "text": text[start + a:start + b]} for k, (a, b) in spans.items()}}
        row["witness_id"] = sha(f"{source_id}\0{version}\0{start}\0{clause}")
        witnesses.append(row)
    return {"schema": SCHEMA, "source_id": source_id, "source_version": version,
            "witnesses": witnesses, "unparsed_source_clauses": skipped,
            "offset_unit": "unicode_character", "prompt_visible": False}


def question_addresses(question: str) -> list[tuple[str, str]]:
    q = canonical(question)
    patterns = [rf"what\s+is\s+(?:the\s+)?(?:current\s+)?(?P<relation>{R})\s+(?:of|for)\s+(?P<entity>{E})",
                rf"what\s+is\s+(?:the\s+)?(?:current\s+)?(?P<entity>{E})(?:'s)?\s+(?P<relation>{R})(?:\s+value)?"]
    for pattern in patterns:
        m = re.fullmatch(pattern, q, re.I)
        if m:
            ent = canonical(m["entity"]).removesuffix("'s")
            if atom(ent):
                return [(ent, ALIASES.get(m["relation"], m["relation"]))]
    m = re.fullmatch(r"what are the (?:current )?(.+) values", q)
    if m:
        out = []
        for part in m[1].split(" and "):
            mm = re.fullmatch(rf"(?P<entity>{E})\s+(?P<relation>{R})", part)
            if not mm or not atom(mm["entity"]):
                return []
            out.append((canonical(mm["entity"]), ALIASES.get(mm["relation"], mm["relation"])))
        return out if len(out) == 2 and len(set(out)) == 2 else []
    return []


def answer_bindings(answer: str, addresses: list[tuple[str, str]]):
    text = normalize_glyphs(answer).strip()
    parsed, unknown = [], []
    for clause, _start in clauses(text):
        result = parse_clause(clause)
        if result:
            parsed.append(result[0])
        else:
            unknown.append(clause)
    if unknown:
        # A scalar answer is an assertion at the ONE question address. This
        # never reads expected answers, source vocabulary, or oracle labels.
        scalar = canonical(text)
        if len(addresses) == 1 and not parsed and len(unknown) == 1 and atom(scalar):
            parsed = [Binding(*addresses[0], scalar)]
            unknown = []
    return parsed, unknown


def verify(answer: str, question: str, mounted: list[dict]) -> dict:
    """Check every assertion, source identity, polarity and join provenance.

    Prior art: Buneman et al. (2001) provenance, Green et al. (2007)
    conjunction. Our receipt lists exact supporting witness IDs; no pooled
    token union, inferred transitive edge, or cross-version binding exists.
    """
    addresses = question_addresses(question)
    bindings, unknown = answer_bindings(answer, addresses)
    result = {"accepted": False, "reason": "", "bindings": [asdict(b) for b in bindings],
              "unknown_clauses": unknown, "supports": [], "joins": [], "prompt_visible": False}
    if not addresses:
        result["reason"] = "unsupported_question"
        return result
    if unknown or not bindings:
        result["reason"] = "unparsed_answer_clause" if unknown else "no_asserted_binding"
        return result
    witnesses, seen = [], {}
    for source in mounted:
        meta = source.get("metadata", {}).get(META)
        if not meta:
            result["reason"] = "missing_deposit_metadata"
            return result
        sid, version = meta["source_id"], meta["source_version"]
        # Re-extraction validates offsets, fields and payload, not merely the
        # hash label. It also rejects mutable or fabricated metadata.
        if (version != sha(source["text"]) or (sid in seen and seen[sid] != version)
                or meta != extract(source["text"], sid, version)):
            result["reason"] = "source_version_or_metadata_mismatch"
            return result
        seen[sid] = version
        witnesses.extend(meta["witnesses"])
    for index, binding in enumerate(bindings):
        address = (binding.entity, binding.relation)
        candidates = [w for w in witnesses if (w["binding"]["entity"], w["binding"]["relation"]) == address]
        matching = [w for w in candidates if w["binding"] == asdict(binding)]
        # Ambiguous affirmative values for a functional slot are not resolved
        # by file order. Version authority must be resolved by repository L2.
        conflict = any((not w["binding"]["negated"] and not binding.negated and w["binding"]["value"] != binding.value)
                       or (w["binding"]["value"] == binding.value and w["binding"]["negated"] != binding.negated) for w in candidates)
        if not matching or conflict:
            result["reason"] = "conflicting_witnesses" if conflict else "no_matching_witness"
            return result
        witness = matching[0]
        result["supports"].append({"assertion_index": index, **{k: witness[k] for k in ("witness_id", "source_id", "source_version", "span")}})
    if not set(addresses) <= {(b.entity, b.relation) for b in bindings}:
        result["reason"] = "question_binding_missing"
        return result
    if len({s["source_id"] for s in result["supports"]}) > 1:
        result["joins"] = [{"kind": "conjunction", "assertion_indices": list(range(len(bindings))),
                            "inputs": result["supports"].copy(), "derived_relation": None}]
    result.update(accepted=True, reason="all_bindings_witnessed")
    return result


def deposit_metadata(graft: dict, source_id: str, source_version: str | None = None):
    if not enabled():
        return None
    meta = extract(graft["text"], source_id, source_version)
    existing = graft.get("metadata", {}).get(META)
    if existing is not None and existing != meta:
        raise ValueError("immutable witness metadata already exists")
    graft.setdefault("metadata", {})[META] = meta
    return meta


def install(arena):
    """Instance-local additive deposit hooks; unset flag is strict identity.

    No class monkeypatch and no prompt hook. Native K/V stays untouched.
    Imported repositories require explicit re-deposit; no on-answer backfill.
    """
    if not enabled():
        return None
    if getattr(arena, "_grm_x2_installed", False):
        return arena
    for name in ("deposit", "deposit_from_cache"):
        original = getattr(arena, name)
        def wrapper(self, text, *args, _original=original, **kwargs):
            index = _original(text, *args, **kwargs)
            deposit_metadata(self.grafts[index], f"graft:{index}")
            return index
        setattr(arena, name, MethodType(wrapper, arena))
    arena._grm_x2_installed = True
    return arena


def gate_answer(answer: str, question: str, mounted: list[dict], info: dict):
    """After existing grounding A, before answer deposit. OFF returns identity.

    The caller's existing grounded bit is required: B = A AND witnesses.
    Structural abstention is a fixed string, never another model generation.
    """
    if not enabled():
        return answer, info
    out = dict(info)
    check = verify(answer, question, mounted)
    out[META] = check
    if not info.get("grounded", False) or not check["accepted"]:
        out.update(abstained=True, abstain_reason="grm_x2_unwitnessed", grounded=False)
        return ABSTENTION, out
    return answer, out
