"""Lead-only native X1 worker. Imported only after a lease and GPU idle check.

Prior art: house EB1/SUP/RT1/RS3/RS4 (2026), native harvest and split routing;
Fisher (1935), matched comparisons (unverified — lead to check).
Borrow native payloads, frame, routing and read path. New: X1 metadata and
controlled duplicate IDs/withheld pages. Duplication shares IMMUTABLE payload
storage, not candidate identity. No direct source text in a serving prompt.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

from scripts.grm_x1_campaign import FIX, SEED, queries, read, registration, score
from scripts.grm_x1_register import create, raw_json
from core.grm_x1_addresses import Address, AddressIndex, AddressedRecall, arena_mount_read, resolve_alias


def clone_node(node):
    # House snapshots (2026): share immutable tensors, copy mutable metadata.
    # Do not deepcopy CUDA tensors. Native host payload is immutable backing.
    shared = {k: v for k, v in node.items() if k in ("h", "host_payload", "cent", "child_cents")}
    copied = deepcopy({k: v for k, v in node.items() if k not in shared and k != "native_node_id"})
    return {**copied, **shared}


def duplicate_children(nodes, parent_id, multiplicity):
    """Exactly m extra identities from the parent's actual native split pages.

    Prior art: RT1 width-child metadata (house 2026); controlled replication
    is standard experimental construction, no prior art known to me for this
    exact X1 stress fixture. Cycle source order, never rank/select by answers.
    """
    parent = nodes[parent_id]
    originals = tuple(parent.get("sources", ()))
    if not (parent.get("metadata") or {}).get("width_guard_parent") or not originals:
        raise ValueError("decoy is not an actual width-guard split family")
    copies = []
    for i in range(multiplicity):
        source = originals[i % len(originals)]
        child = clone_node(nodes[source])
        idx = len(nodes)
        child["node_id"] = idx
        child["metadata"] = {**child.get("metadata", {}), "width_guard_child": True,
                              "x1_duplicate_of": source, "culled_from": parent_id,
                              "active": True}
        child["sources"] = [parent_id]
        child["retired"] = False
        nodes.append(child)
        copies.append(idx)
    parent["sources"] = [*originals, *copies]
    parent["metadata"]["width_guard_children"] = list(parent["sources"])
    parent["child_cents"] = [nodes[i]["cent"] for i in parent["sources"]]
    return copies


def new_repo(model, tokenizer, e2e, path, frame):
    from core.gpt_oss20b_tc import gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    repo = GraftRepository(model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False), str(path),
        autosave=False, native_auto=False, native_lib_path=frame["native_lib"],
        arena_cls=e2e.GptOssGQAArenaCache, vram_budget_mb=None,
        route_layer=int(gpt_oss_grm_dialect_kwargs(model.config)["route_layer"]),
        arena_width=96, topk=3, live_turns=2, max_live=4096, ephemeral=True,
        sink_text=e2e.HARMONY_SINK, prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS, storage_bits=8,
        revision_resolution=True, decisive_admission=True)
    repo.arena._rs3_capture_pin_explicit = "live"
    repo.arena._rs3_seat_explicit = True
    return repo


def install(repo, e2e, family):
    """SUP's chronological native deposit/lineage shape, tagged at deposit.

    Fixture oracle tags version and entity/relation; labels never derive from
    question text. Publish all current source pages (no cherry-picked span).
    """
    arena = repo.arena
    ids, index = {}, AddressIndex(family["family_id"])
    for node in family["nodes"]:
        user, assistant = node["text"].split("\nAssistant: ", 1)
        text = e2e.harmony_turn(user.removeprefix("User: "), assistant)
        idx = arena.deposit(text, capture_pin="live")
        ids[node["node_id"]] = idx
        arena.grafts[idx].update({"node_id": idx, "kind": "fact", "metadata": {
            "kind": "fact", "active": True, "supersedes": [], "superseded_by": [],
            "x1_deposit_address": node["address"], "x1_deposit_version": node["version"]}})
    for node in family["nodes"]:
        idx = ids[node["node_id"]]
        older = [ids[name] for name in node["supersedes"]]
        arena.grafts[idx]["metadata"]["supersedes"] = older
        for old in older:
            arena.grafts[old]["metadata"]["superseded_by"].append(idx)
    arena._bump_cuda_gqa_epoch()
    for idx in range(len(arena.grafts)):
        repo._native_sync_node(idx)
    # Existing production width guard; duplicates are ACTUAL children of it.
    for node in family["nodes"]:
        repo._guard_deposit_width(ids[node["node_id"]])
    for node in family["nodes"]:
        idx = ids[node["node_id"]]
        graft = arena.grafts[idx]
        pages = list(graft["sources"]) if graft["metadata"].get("width_guard_parent") else [idx]
        index.publish(Address(*node["address"]), node["version"], pages,
                      hashlib.sha256(node["text"].encode()).hexdigest())
    return ids, index


def payload_digest(repo, idx):
    """Hash the packed backing used by native sync and the repository loader.

    Prior art: GraftRepository (house, 2026), _native_sync_node and
    _ensure_host_payload/_read_payload_file; verified in local source.
    Reuse its RAM-first/durable-file accessor and existing X1 SHA-256
    key/shape/dtype/bytes hashing. New: resolve backing before X1 snapshots;
    no new hashing or paging algorithm. These are packed-payload digests,
    not the former device-h digests, and must not be compared across r1/r2.
    """
    import numpy as np
    node = repo.arena.grafts[idx]
    repo._ensure_host_payload(idx, node)
    payload = node["host_payload"]
    if not payload:
        raise RuntimeError(f"graft {idx} has no packed payload to hash")
    digest = hashlib.sha256()
    for key in sorted(payload):
        array = np.asarray(payload[key])
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_cell(cell, rows, directory):
    import time
    import numpy as np
    import tensor_cuda as tc
    from scripts import grm_e2e_session as e2e
    from core import kv_graft
    from core.gpt_oss20b_tc import GptOss20B_TC
    from transformers import AutoTokenizer

    frame = registration()["frame"]
    # Explicit registered serving flags in this worker only, never live config.
    os.environ.update({"GRM_LSR_FIXES": "1", "GRM_DEMAND_NGH": "0", "GRM_PERSISTENT_BOAT": "0",
                       "GRM_CAPTURE_PIN": "live", "GRM_SEAT_NEAR_LIVE": "1", "GRM_SUP_RESOLVE": "1",
                       "GRM_ADM_DECISIVE": "1", "GRM_PROBE_LADDER": "1"})
    model, model_info = GptOss20B_TC.from_pretrained(frame["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(frame["model_dir"], local_files_only=True)
    directory.mkdir(parents=True, exist_ok=False)
    create(directory / "model.json", raw_json({"model_id": frame["model_id"],
        "reasoning_effort": frame["reasoning_effort"], "model_info": str(model_info), "frame": frame}))
    query_rows = queries()
    families, aliases = read(FIX / "families.json"), read(FIX / "aliases.json")
    snapshots = {}
    try:
        for ordinal, query in enumerate(query_rows):
            if query["query_id"] not in cell["query_ids"]:
                continue
            family = families[query["family_id"]]
            if query["family_id"] not in snapshots:
                template = new_repo(model, tokenizer, e2e, directory / (query["family_id"] + "_capture"), frame)
                try:
                    ids, index = install(template, e2e, family)
                    # Resolve cold backing in its owning repository BEFORE
                    # cloning/closing it; private arm repos share these arrays.
                    hashes = {i: payload_digest(template, i)
                              for i in range(len(template.arena.grafts))}
                    nodes = [clone_node(g) for g in template.arena.grafts]
                    target_pages = index.resolve(Address(*family["address"])).page_ids
                    if sum(nodes[i]["ntok"] for i in target_pages) > 96:
                        raise RuntimeError("oracle current source pages exceed width 96; no cherry-picking permitted")
                    duplicate_ids = duplicate_children(nodes, ids[family["decoy_node"]], cell["multiplicity"])
                    # Captured hashes identify shared payloads and the stress
                    # inventory separately. Hashing is excluded from turn wall.
                    hashes.update({i: hashes[nodes[i]["metadata"]["x1_duplicate_of"]] for i in duplicate_ids})
                    snapshots[query["family_id"]] = nodes, index.dump(), target_pages, hashes
                    index.save(directory / (query["family_id"] + "_addresses.json"))
                    create(directory / (query["family_id"] + "_pages.json"), raw_json({
                        "target_pages": target_pages, "duplicate_ids": duplicate_ids,
                        "payload_digest_format": "packed-host-key-shape-dtype-bytes-v1",
                        "page_payload_shas": hashes, "seed": SEED,
                        "duplicate_metadata": {i: nodes[i]["metadata"] for i in duplicate_ids}}))
                finally:
                    template.close()
                    kv_graft.clear_injection(model)
            nodes, index_dump, target_pages, hashes = snapshots[query["family_id"]]
            for condition_index, condition in enumerate(cell["conditions"]):
                arms = list(cell["arms"])
                rotation = (ordinal + condition_index) % len(arms)
                arms = arms[rotation:] + arms[:rotation]
                for arm in arms:
                    # New private repository/cache per arm; underlying source
                    # payload arrays are shared IMMUTABLE objects. No A->B state.
                    random.seed(SEED + ordinal)
                    np.random.seed(SEED + ordinal)
                    turn_path = directory / f"{query['query_id']}_{condition}_{arm}"
                    repo = new_repo(model, tokenizer, e2e, turn_path, frame)
                    try:
                        arena = repo.arena
                        arena.grafts = [clone_node(g) for g in nodes]
                        for idx, graft in enumerate(arena.grafts):
                            if condition == "absent" and idx in target_pages:
                                graft["retired"] = True
                                graft.setdefault("metadata", {})["active"] = False
                            repo._native_sync_node(idx)
                        arena._bump_cuda_gqa_epoch()
                        if arena.width != 96 or not arena.ephemeral or arena.storage_bits != 8:
                            raise RuntimeError("live frame does not match registered frame")
                        index = AddressIndex.restore(index_dump, repository_id=family["family_id"])
                        available = {i for i, g in enumerate(arena.grafts) if not g.get("retired")
                                     and g.get("metadata", {}).get("active", True)}
                        fallback = lambda question, **kw: arena.step(question, deposit=False, defer_memory=True, demand_ngh=False, **kw)
                        recall = AddressedRecall(index, fallback, partial(arena_mount_read, arena), lambda: available,
                                                 environ={"GRM_X1_ADDRESSES": "0" if arm == "A" else "1"})
                        # Resolve naturally INSIDE wall only for the gated N arm.
                        kv_graft.clear_injection(model)
                        arena.caches, arena.pos, arena.live_segs = None, 0, []
                        arena.cur_mounts, arena.cur_mount_n = [], 0
                        tc.synchronize()
                        started = time.perf_counter()
                        address = resolve_alias(query["question"], aliases) if arm == "N" else Address(*query["oracle_address"])
                        answer, info = recall.serve(query["question"], address=address, point_lookup=True,
                                                   page_fault=arm in ("C", "N"), ngen=frame["ngen"], max_trips=frame["max_trips"])
                        tc.synchronize()
                        wall = time.perf_counter() - started
                        actual = [int(v) for v in arena.cur_mounts]
                        if condition == "absent" and set(actual) & set(target_pages):
                            raise RuntimeError("absence intervention failed: current-version page mounted")
                        selected_info = {k: v for k, v in info.items() if k.startswith(("x1_", "demand_", "abstain", "frame_", "fit_", "admission_", "rt1_", "recency_", "live_segments_"))}
                        row = {"evidence_class": "E2E session receipt", "query_id": query["query_id"],
                            "kind": query["kind"], "family_id": query["family_id"], "multiplicity": cell["multiplicity"],
                            "arm": arm, "condition": condition, "seed": SEED + ordinal,
                            "question": query["question"], "answer": str(answer), "wall_s": wall,
                            "expected": query["expected"], "mounted_ids": actual, "target_page_ids": target_pages,
                            "address_coverage": tuple(actual) == tuple(target_pages),
                            "generation_calls": info.get("x1_generation_calls", 0 if info.get("abstained") else None),
                            "resolved_address": None if address is None else [address.entity, address.relation],
                            "info": selected_info, **score(answer, query, condition, info)}
                        create(turn_path / "turn_receipt.json", raw_json(row))
                        rows.append(row)
                        print(json.dumps({k: row[k] for k in ("query_id", "condition", "arm", "exact_answer", "false_answer", "wall_s")}), flush=True)
                    finally:
                        repo.close()
                        kv_graft.clear_injection(model)
    finally:
        snapshots.clear()
        kv_graft.clear_injection(model)
